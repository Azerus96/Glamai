import os
import gradio as gr
import requests
import json
import time
import base64
import tempfile
import shutil
import threading
import uuid
import sys
from pathlib import Path

# Настройка логирования
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("glama-chat")

# Проверка наличия API ключа
api_key = os.environ.get("GLAMA_API_KEY")
if not api_key:
    logger.error("API ключ не найден. Установите переменную окружения GLAMA_API_KEY.")
    raise ValueError("API ключ не найден. Установите переменную окружения GLAMA_API_KEY.")

logger.info(f"API ключ найден (первые 5 символов: {api_key[:5]}...)")

# Создание временной директории для файлов
TEMP_DIR = Path(tempfile.gettempdir()) / "glama_chat_files"
TEMP_DIR.mkdir(exist_ok=True)
logger.info(f"Временная директория создана: {TEMP_DIR}")

# Максимальный размер файла (10 МБ)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Кэш для моделей
models_cache = {
    "timestamp": 0,
    "models": ["gpt-4.5-preview-2025-02-27", "openai/gpt-4.5-preview-2025-02-27", "anthropic/claude-3-opus-20240229"]
}

# Функция для получения списка доступных моделей с кэшированием
def get_available_models(force_refresh=False):
    current_time = time.time()
    # Используем кэш, если он не старше 1 часа и не требуется принудительное обновление
    if not force_refresh and current_time - models_cache["timestamp"] < 3600:
        logger.info(f"Используем кэшированный список моделей: {len(models_cache['models'])} моделей")
        return models_cache["models"]
    
    try:
        logger.info("Запрашиваем список моделей от API...")
        # Запрос к API для получения списка моделей
        response = requests.get(
            "https://glama.ai/api/gateway/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10  # Таймаут 10 секунд
        )
        
        if response.status_code == 200:
            models_data = response.json()
            # Извлекаем только ID моделей и сортируем их
            model_ids = [model["id"] for model in models_data["data"]]
            model_ids.sort()
            
            # Обновляем кэш
            models_cache["timestamp"] = current_time
            models_cache["models"] = model_ids
            
            logger.info(f"Получен список моделей: {len(model_ids)} моделей")
            return model_ids
        else:
            logger.error(f"Ошибка при получении списка моделей: {response.status_code}")
            return models_cache["models"]
    except Exception as e:
        logger.exception(f"Исключение при получении списка моделей: {str(e)}")
        return models_cache["models"]

# Простая функция для отправки сообщения в API без потоковой обработки
def send_simple_message(message, model_name="gpt-4.5-preview-2025-02-27"):
    try:
        logger.info(f"Отправка тестового сообщения к API. Модель: {model_name}")
        
        # Подготовка данных для запроса
        request_data = {
            "model": model_name,
            "messages": [{"role": "user", "content": message}],
            "temperature": 0.7,
            "stream": False
        }
        
        # Отправка запроса к API
        response = requests.post(
            "https://glama.ai/api/gateway/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("Получен успешный ответ от API")
            return result["choices"][0]["message"]["content"]
        else:
            logger.error(f"Ошибка API: {response.status_code}")
            try:
                error_json = response.json()
                return f"Ошибка API: {response.status_code} - {error_json.get('error', {}).get('message', 'Неизвестная ошибка')}"
            except:
                return f"Ошибка API: {response.status_code}"
    except Exception as e:
        logger.exception(f"Ошибка при отправке тестового сообщения: {str(e)}")
        return f"Ошибка: {str(e)}"

# Функция для обработки сообщений
def process_message(message, history, files, model_name, temperature, max_tokens, system_message):
    # Создаем уникальный ID для запроса
    request_id = str(uuid.uuid4())
    logger.info(f"Новый запрос {request_id}. Модель: {model_name}, Температура: {temperature}")
    
    try:
        # Подготовка сообщений из истории
        messages = []
        
        # Добавляем системное сообщение, если оно есть
        if system_message:
            messages.append({"role": "system", "content": system_message})
            logger.info(f"Добавлено системное сообщение: {system_message[:50]}...")
        
        if history:
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
            logger.info(f"Добавлена история: {len(history)} сообщений")
        
        # Добавление текущего сообщения (упрощенно, только текст)
        messages.append({"role": "user", "content": message})
        logger.info(f"Добавлено текущее сообщение: {message[:50]}...")
        
        # Подготовка данных для запроса
        request_data = {
            "model": model_name,
            "messages": messages,
            "temperature": float(temperature),
            "stream": True
        }
        
        if max_tokens:
            request_data["max_tokens"] = int(max_tokens)
        
        # Отправка запроса к API с использованием requests
        logger.info("Отправка запроса к API...")
        response = requests.post(
            "https://glama.ai/api/gateway/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json=request_data,
            stream=True,
            timeout=60  # Увеличиваем таймаут
        )
        
        # Проверка статуса ответа
        if response.status_code != 200:
            error_msg = f"Ошибка API: {response.status_code}"
            logger.error(error_msg)
            try:
                error_json = response.json()
                error_msg += f" - {error_json.get('error', {}).get('message', 'Неизвестная ошибка')}"
            except:
                pass
            return error_msg
        
        logger.info("Начинаем обработку потокового ответа...")
        
        # Обработка потокового ответа
        response_text = ""
        for line in response.iter_lines():
            if line:
                try:
                    line = line.decode('utf-8')
                    logger.debug(f"Получена строка от API: {line[:50]}...")
                    
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            logger.info("Получен маркер завершения [DONE]")
                            break
                        
                        try:
                            chunk = json.loads(data)
                            if chunk.get('choices') and chunk['choices'][0].get('delta') and chunk['choices'][0]['delta'].get('content'):
                                content = chunk['choices'][0]['delta']['content']
                                response_text += content
                                logger.debug(f"Добавлен текст: {content[:20]}...")
                                yield response_text
                        except json.JSONDecodeError as e:
                            logger.error(f"Ошибка декодирования JSON: {str(e)}")
                except Exception as e:
                    logger.exception(f"Ошибка при обработке строки: {str(e)}")
        
        if not response_text:
            logger.warning("Не получен ответ от API")
            # Пробуем отправить простой запрос для проверки API
            test_response = send_simple_message("Тестовое сообщение", model_name)
            if test_response.startswith("Ошибка"):
                return "Не получен ответ от API. Проверьте настройки и API-ключ."
            else:
                return "Не получен потоковый ответ от API, но API работает. Попробуйте еще раз."
        
        logger.info(f"Ответ получен, длина: {len(response_text)} символов")
        return response_text
    except requests.exceptions.RequestException as e:
        error_msg = f"Ошибка сетевого запроса: {str(e)}"
        logger.exception(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"Ошибка при обращении к API: {str(e)}"
        logger.exception(error_msg)
        return error_msg

# Получаем список моделей при запуске
logger.info("Получение списка моделей при запуске...")
available_models = get_available_models()
default_model = available_models[0] if available_models else "gpt-4.5-preview-2025-02-27"
logger.info(f"Выбрана модель по умолчанию: {default_model}")

# Создание интерфейса
with gr.Blocks(theme=gr.themes.Soft(), css="""
    footer {visibility: hidden}
    .generating {
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% {opacity: 1;}
        50% {opacity: 0.5;}
        100% {opacity: 1;}
    }
""") as demo:
    gr.Markdown("# Чат с LLM через Glama Gateway")
    
    with gr.Row():
        with gr.Column(scale=3):
            model_dropdown = gr.Dropdown(
                choices=available_models,
                label="Модель",
                value=default_model,
                info="Доступные модели языкового интеллекта"
            )
            refresh_models_btn = gr.Button("Обновить список моделей", size="sm")
    
    with gr.Accordion("Настройки", open=False):
        system_message = gr.Textbox(
            label="Системное сообщение",
            placeholder="Введите системное сообщение для модели...",
            lines=2
        )
        
        with gr.Row():
            with gr.Column():
                temperature_slider = gr.Slider(
                    minimum=0.0, 
                    maximum=2.0, 
                    value=0.7, 
                    step=0.1, 
                    label="Температура",
                    info="Контролирует случайность ответов (0 = детерминированные, 2 = максимально случайные)"
                )
            with gr.Column():
                max_tokens_slider = gr.Slider(
                    minimum=100, 
                    maximum=8000, 
                    value=4000, 
                    step=100, 
                    label="Максимальное количество токенов",
                    info="Ограничивает длину ответа модели"
                )
    
    chatbot = gr.Chatbot(height=500, show_copy_button=True)
    
    with gr.Row():
        with gr.Column(scale=8):
            msg = gr.Textbox(
                show_label=False,
                placeholder="Введите сообщение...",
                container=False,
                lines=3
            )
        with gr.Column(scale=1, min_width=50):
            submit_btn = gr.Button("Отправить", variant="primary")
    
    file_upload = gr.File(label="Прикрепить файл")
    
    with gr.Row():
        clear = gr.Button("Очистить чат")
        test_api_btn = gr.Button("Проверить API")
    
    # Индикатор состояния
    status_indicator = gr.Markdown("Готов к работе")
    
    # Функция для обработки сообщений
    def user_input(message, chat_history, file, model_name, temperature, max_tokens, system_msg):
        logger.info(f"Получено сообщение: {message[:50]}...")
        
        # Проверка на None
        if chat_history is None:
            chat_history = []
            logger.info("История чата была None, создан пустой список")
            
        if not message:
            logger.warning("Получено пустое сообщение")
            return "", chat_history, None, "Введите сообщение"
        
        # Обновляем статус
        logger.info("Обработка сообщения...")
        yield "", chat_history, None, "Обработка сообщения..."
        
        # Добавляем сообщение пользователя в историю
        chat_history.append((message, None))
        logger.info("Сообщение добавлено в историю")
        yield "", chat_history, None, "Генерация ответа..."
        
        # Получаем ответ от модели
        bot_message = ""
        try:
            logger.info("Запуск генерации ответа...")
            for response in process_message(message, chat_history[:-1], file, model_name, temperature, max_tokens, system_msg):
                bot_message = response
                chat_history[-1] = (message, bot_message)
                logger.debug(f"Получен частичный ответ, длина: {len(bot_message)} символов")
                yield "", chat_history, None, "Генерация ответа..."
        except Exception as e:
            logger.exception(f"Ошибка при генерации ответа: {str(e)}")
            bot_message = f"Ошибка: {str(e)}"
            chat_history[-1] = (message, bot_message)
            yield "", chat_history, None, "Произошла ошибка"
        
        logger.info("Генерация ответа завершена")
        return "", chat_history, None, "Готов к работе"
    
    # Функция для обновления списка моделей
    def refresh_models():
        logger.info("Обновление списка моделей...")
        new_models = get_available_models(force_refresh=True)
        logger.info(f"Получено {len(new_models)} моделей")
        return gr.Dropdown(choices=new_models, value=new_models[0] if new_models else None), "Список моделей обновлен"
    
    # Функция для проверки API
    def test_api():
        logger.info("Проверка API...")
        try:
            response = send_simple_message("Привет! Это тестовое сообщение для проверки API.")
            if response.startswith("Ошибка"):
                logger.error(f"Ошибка при проверке API: {response}")
                return "API недоступен: " + response
            else:
                logger.info("API работает корректно")
                return "API работает корректно. Ответ: " + response[:100] + "..."
        except Exception as e:
            logger.exception(f"Исключение при проверке API: {str(e)}")
            return f"Ошибка при проверке API: {str(e)}"
    
    # Функция для очистки чата
    def clear_chat():
        logger.info("Очистка чата")
        return [], None, "Чат очищен"
    
    # Привязка событий
    msg.submit(user_input, [msg, chatbot, file_upload, model_dropdown, temperature_slider, max_tokens_slider, system_message], 
               [msg, chatbot, file_upload, status_indicator])
    
    submit_btn.click(user_input, [msg, chatbot, file_upload, model_dropdown, temperature_slider, max_tokens_slider, system_message], 
                    [msg, chatbot, file_upload, status_indicator])
    
    clear.click(clear_chat, outputs=[chatbot, file_upload, status_indicator])
    
    test_api_btn.click(test_api, outputs=[status_indicator])
    
    refresh_models_btn.click(refresh_models, outputs=[model_dropdown, status_indicator])

# Запуск приложения
if __name__ == "__main__":
    logger.info("Запуск приложения...")
    port = int(os.environ.get("PORT", 7860))
    logger.info(f"Используется порт: {port}")
    demo.launch(server_name="0.0.0.0", server_port=port)
