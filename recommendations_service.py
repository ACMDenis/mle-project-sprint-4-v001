import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Music Recommendation Service", version="1.0.0")

# Пути к файлам данных

TOP_POPULAR_PATH = ("/home/mle-user/mle-project-sprint-4-v001/top_popular.parquet")
PERSONAL_RECS_PATH = ("/home/mle-user/mle-project-sprint-4-v001/final_recommendations.parquet")
SIMILAR_TRACKS_PATH = ("/home/mle-user/mle-project-sprint-4-v001/similar.parquet")

# Загрузка данных
def load_data():
    try:
        top_popular = pd.read_parquet(TOP_POPULAR_PATH)
        if 'track_id' in top_popular.columns:
            top_popular = top_popular['track_id'].head(10).tolist()
        else:
            logger.error("top_popular.parquet doesn't have 'track_id' column")
            top_popular = []
        logger.info(f"Loaded top popular: {len(top_popular)} tracks")
    except Exception as e:
        logger.error(f"Error loading top popular: {e}")
        top_popular = []

    try:
        personal_recs = pd.read_parquet(PERSONAL_RECS_PATH)
        logger.info(f"Loaded personal recs: {personal_recs.shape[0]} recommendations")
    except Exception as e:
        logger.error(f"Error loading personal recs: {e}")
        personal_recs = pd.DataFrame(columns=['user_id', 'track_id', 'lgb_score'])
    
    try:
        similar_tracks = pd.read_parquet(SIMILAR_TRACKS_PATH)
        logger.info(f"Loaded similar tracks: {similar_tracks.shape[0]} pairs")
    except Exception as e:
        logger.error(f"Error loading similar tracks: {e}")
        similar_tracks = pd.DataFrame(columns=['original_track_id', 'similar_track_id', 'score'])
    
    return top_popular, personal_recs, similar_tracks

top_popular, personal_recs, similar_tracks = load_data()

online_history = {}

class RecommendationRequest(BaseModel):
    user_id: int
    online_history: Optional[List[int]] = None

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, object]]

@app.get("/")
async def root():
    return {
        "service": "Music Recommendations",
        "version": app.version,
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend (POST)"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {
        "status": "OK",
        "details": {
            "top_popular_count": len(top_popular),
            "personal_recs_count": len(personal_recs),
            "similar_pairs_count": len(similar_tracks)
        }
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    user_id = request.user_id
    logger.info(f"Recommendation request for user: {user_id}")
    
    # Обновляем онлайн-историю, если предоставлена
    if request.online_history:
        online_history[user_id] = request.online_history
        logger.info(f"Updated online history for user {user_id}: {request.online_history[:3]}...")
    
    # Получаем офлайн-рекомендации
    offline_recs = get_offline_recommendations(user_id)
    logger.info(f"Offline recs for user {user_id}: {len(offline_recs)} items")
    
    # Получаем онлайн-рекомендации на основе истории
    online_recs = get_online_recommendations(user_id)
    logger.info(f"Online recs for user {user_id}: {len(online_recs)} items")
    
    # Смешиваем рекомендации
    recommendations = blend_recommendations(offline_recs, online_recs)
    logger.info(f"Blended recs for user {user_id}: {len(recommendations)} items")
    
    return {
        "user_id": user_id,
        "recommendations": recommendations
    }

def get_offline_recommendations(user_id: int) -> List[Dict]:
    """Получение офлайн-рекомендаций"""
    if not personal_recs.empty and 'user_id' in personal_recs.columns:
        user_recs = personal_recs[personal_recs['user_id'] == user_id]
        if not user_recs.empty:
            # Проверка наличия необходимых колонок
            required_columns = ['track_id', 'lgb_score']
            if all(col in user_recs.columns for col in required_columns):
                return user_recs.sort_values('lgb_score', ascending=False)\
                                .head(10)[['track_id', 'lgb_score']]\
                                .rename(columns={'lgb_score': 'score'})\
                                .to_dict('records')
    
    # Возвращаем топ популярных, если нет персональных рекомендаций
    if top_popular:
        return [{'track_id': track, 'score': 1.0} for track in top_popular[:10]]
    return []

def get_online_recommendations(user_id: int) -> List[Dict]:
    """Генерация онлайн-рекомендаций на основе истории"""
    if user_id not in online_history or not online_history[user_id]:
        return []
    
    # Проверка наличия необходимых колонок в similar_tracks
    required_columns = ['original_track_id', 'similar_track_id', 'score']
    if not all(col in similar_tracks.columns for col in required_columns):
        return []
    
    # Получаем похожие треки для каждого трека в истории
    recommendations = {}
    for track_id in online_history[user_id][:10]:  # Ограничим историю
        similar = similar_tracks[similar_tracks['original_track_id'] == track_id]
        if not similar.empty:
            for _, row in similar.iterrows():
                sim_track = row['similar_track_id']
                score = row['score']
                
                # Обновляем счетчик и максимальный score
                if sim_track in recommendations:
                    recommendations[sim_track]['count'] += 1
                    recommendations[sim_track]['max_score'] = max(
                        recommendations[sim_track]['max_score'], score
                    )
                else:
                    recommendations[sim_track] = {
                        'track_id': sim_track,
                        'max_score': score,
                        'count': 1
                    }
    
    # Преобразуем в список и ранжируем
    recs_list = list(recommendations.values())
    recs_list.sort(key=lambda x: (x['count'], x['max_score']), reverse=True)
    
    return [{
        'track_id': item['track_id'],
        'score': item['max_score']
    } for item in recs_list[:10]]

def blend_recommendations(
    offline: List[Dict], 
    online: List[Dict]
) -> List[Dict]:
    """Смешивание онлайн- и офлайн-рекомендаций"""
    # Создаем множество уже рекомендованных треков
    seen = set()
    blended = []
    
    # Добавляем онлайн-рекомендации (первые 3)
    for rec in online[:3]:
        track_id = rec['track_id']
        if track_id not in seen:
            rec['source'] = 'online'
            blended.append(rec)
            seen.add(track_id)
    
    # Добавляем офлайн-рекомендации
    for rec in offline:
        track_id = rec['track_id']
        if track_id not in seen and len(blended) < 10:
            rec['source'] = 'offline'
            blended.append(rec)
            seen.add(track_id)
    
    # Если остались места, добавляем топ популярных
    if len(blended) < 10 and top_popular:
        for track in top_popular:
            if track not in seen and len(blended) < 10:
                blended.append({
                    'track_id': track,
                    'score': 1.0,
                    'source': 'popular'
                })
                seen.add(track)
    
    return blended[:10]  # Гарантируем не более 10 рекомендаций

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("recommendations_service:app", host="0.0.0.0", port=8000, reload=True)