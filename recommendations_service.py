import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI()

# Загрузка данных
try:
    top_popular = pd.read_parquet('/home/mle-user/mle-project-sprint-4-v001/top_popular.parquet')['track_id'].head(10).tolist()
    personal_recs = pd.read_parquet('/home/mle-user/mle-project-sprint-4-v001/final_recommendations.parquet')
    similar_tracks = pd.read_parquet('/home/mle-user/mle-project-sprint-4-v001/similar.parquet')
except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    top_popular = []
    personal_recs = pd.DataFrame(columns=['user_id', 'track_id', 'lgb_score'])
    similar_tracks = pd.DataFrame(columns=['original_track_id', 'similar_track_id', 'score'])

# Хранилище онлайн-истории (в реальной системе лучше использовать Redis)
online_history = {}

class RecommendationRequest(BaseModel):
    user_id: int
    online_history: Optional[List[int]] = None

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, object]]

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    user_id = request.user_id
    
    # Обновляем онлайн-историю, если предоставлена
    if request.online_history:
        online_history[user_id] = request.online_history
    
    # Получаем офлайн-рекомендации
    offline_recs = get_offline_recommendations(user_id)
    
    # Получаем онлайн-рекомендации на основе истории
    online_recs = get_online_recommendations(user_id)
    
    # Смешиваем рекомендации
    recommendations = blend_recommendations(offline_recs, online_recs)
    
    return {
        "user_id": user_id,
        "recommendations": recommendations
    }

def get_offline_recommendations(user_id: int) -> List[Dict]:
    """Получение офлайн-рекомендаций"""
    if not personal_recs.empty:
        user_recs = personal_recs[personal_recs['user_id'] == user_id]
        if not user_recs.empty:
            return user_recs.sort_values('lgb_score', ascending=False)\
                            .head(10)[['track_id', 'lgb_score']]\
                            .rename(columns={'lgb_score': 'score'})\
                            .to_dict('records')
    
    # Возвращаем топ популярных, если нет персональных рекомендаций
    return [{'track_id': track, 'score': 1.0, 'source': 'popular'} 
            for track in top_popular[:10]]

def get_online_recommendations(user_id: int) -> List[Dict]:
    """Генерация онлайн-рекомендаций на основе истории"""
    if user_id not in online_history:
        return []
    
    # Получаем похожие треки для каждого трека в истории
    recommendations = {}
    for track_id in online_history[user_id]:
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
        'score': item['max_score'],
        'source': 'online'
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
        if rec['track_id'] not in seen:
            rec['source'] = 'online'
            blended.append(rec)
            seen.add(rec['track_id'])
    
    # Добавляем офлайн-рекомендации
    for rec in offline:
        if rec['track_id'] not in seen and len(blended) < 10:
            rec['source'] = 'offline'
            blended.append(rec)
            seen.add(rec['track_id'])
    
    # Если остались места, добавляем топ популярных
    if len(blended) < 10:
        for track in top_popular:
            if track not in seen and len(blended) < 10:
                blended.append({
                    'track_id': track,
                    'score': 1.0,
                    'source': 'popular'
                })
                seen.add(track)
    
    return blended

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)