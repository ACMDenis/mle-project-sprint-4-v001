import requests
import time
import logging
import os

# Настройка логирования
logging.basicConfig(
    filename='test_service.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

SERVICE_URL = "http://localhost:8000"
RECOMMEND_ENDPOINT = f"{SERVICE_URL}/recommend"
HEALTH_ENDPOINT = f"{SERVICE_URL}/health"

def check_service_health():
    """Проверка доступности сервиса"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            logging.info("Service is healthy")
            return True
        logging.error(f"Service health check failed: {response.status_code}")
        return False
    except Exception as e:
        logging.error(f"Service connection failed: {str(e)}")
        return False

def test_recommendations(user_id, online_history=None, test_name=""):
    payload = {"user_id": user_id}
    if online_history:
        payload["online_history"] = online_history
    
    try:
        start_time = time.time()
        response = requests.post(RECOMMEND_ENDPOINT, json=payload, timeout=10)
        duration = time.time() - start_time
        
        if response.status_code != 200:
            logging.error(f"{test_name}: Error {response.status_code} - {response.text}")
            return None
        
        data = response.json()
        logging.info(f"{test_name}: Success")
        logging.info(f"User ID: {user_id}")
        logging.info(f"Recommendations count: {len(data['recommendations'])}")
        logging.info(f"Response time: {duration:.3f}s")
        
        # Анализ источников рекомендаций
        sources = {}
        for rec in data['recommendations']:
            source = rec.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        logging.info("Sources distribution:")
        for source, count in sources.items():
            logging.info(f"  {source}: {count} items")
        
        return data
    
    except Exception as e:
        logging.exception(f"{test_name}: Test failed with exception")
        return None

def run_tests():
    """Запуск всех тестов"""
    logging.info("="*60)
    logging.info("Starting recommendation service tests")
    logging.info("="*60)
    
    # Проверка здоровья сервиса
    logging.info("Checking service health...")
    if not check_service_health():
        logging.error("Service is not available. Exiting tests.")
        return
    
    # Тест 1: Пользователь без персональных рекомендаций
    logging.info("\n" + "="*60)
    logging.info("Test 1: New user (no personal recommendations)")
    logging.info("="*60)
    test_recommendations(
        user_id=999999999,
        test_name="New user test"
    )
    
    # Тест 2: Пользователь с персональными рекомендациями, но без онлайн-истории
    logging.info("\n" + "="*60)
    logging.info("Test 2: User with personal recommendations but no online history")
    logging.info("="*60)
    test_user_id = 12345
    test_recommendations(
        user_id=test_user_id,
        test_name="Existing user without history"
    )
    
    # Тест 3: Пользователь с персональными рекомендациями и онлайн-историей
    logging.info("\n" + "="*60)
    logging.info("Test 3: User with personal recommendations and online history")
    logging.info("="*60)
    test_recommendations(
        user_id=test_user_id,
        online_history=[789, 456, 123],
        test_name="User with online history"
    )
    
    logging.info("\n" + "="*60)
    logging.info("Testing completed")

if __name__ == "__main__":
    run_tests()