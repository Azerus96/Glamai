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
from pathlib import Path

# Проверка наличия API ключа
api_key = os.environ.get("GLAMA_API_KEY")
if not api_key:
    raise ValueError("API ключ не найден. Установите переменную окружения GLAMA_API_KEY.")

# Создание временной директории для файлов
TEMP_DIR = Path(tempfile.gettempdir()) / "glama_chat_files"
TEMP_DIR.mkdir(exist_ok=True)

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
        return models_cache["models"]
    
    try:
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
            
            return model_ids
        else:
            print(f"Ошибка при получении списка моделей: {response.status_code}")
            return models_cache["models"]
    except Exception as e:
        print(f"Исключение при получении списка моделей: {str(e)}")
        return models_cache["models"]

# Функция для обработки файлов
def process_files(files):
    file_contents = []
    file_paths = []
    
    if not files:
        return file_contents, file_paths
    
    for file in files:
        # Проверка размера файла
        if os.path.getsize(file.name) > MAX_FILE_SIZE:
            file_contents.append(f"Файл {file.name} слишком большой (максимум 10 МБ).")
            continue
        
        # Создаем уникальное имя для временного файла
        file_ext = os.path.splitext(file.name)[1]
        temp_file = TEMP_DIR / f"{uuid.uuid4()}{file_ext}"
        
        # Копируем файл во временную директорию
        shutil.copy2(file.name, temp_file)
        file_paths.append(temp_file)
        
        # Определение типа файла
        file_type = "image" if file.name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')) else "file"
        
        if file_type == "image":
            # Для изображений не добавляем содержимое в текст
            pass
        else:
            # Для текстовых файлов читаем содержимое
            try:
                with open(file.name, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    file_contents.append(f"Содержимое файла {os.path.basename(file.name)}:\n{file_content}")
            except UnicodeDecodeError:
                try:
                    # Пробуем другую кодировку
                    with open(file.name, 'r', encoding='latin-1') as f:
                        file_content = f.read()
                        file_contents.append(f"Содержимое файла {os.path.basename(file.name)}:\n{file_content}")
                except:
                    file_contents.append(f"Файл {os.path.basename(file.name)} не может быть прочитан как текст.")
    
    return file_contents, file_paths

# Функция для подготовки содержимого сообщения
def prepare_message_content(message, files):
    content = []
    
    # Добавление текста сообщения
    if message:
        content.append({"type": "text", "text": message})
    
    # Обработка файлов, если они есть
    file_contents = []
    if files:
        file_contents, file_paths = process_files(files)
        
        # Добавление изображений
        for file_path in file_paths:
            if str(file_path).lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                try:
                    with open(file_path, 'rb') as img_file:
                        base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                        ext = os.path.splitext(file_path)[1][1:]  # Получаем расширение без точки
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{ext};base64,{base64_image}"
                            }
                        })
                except Exception as e:
                    if message:
                        content[0]["text"] += f"\n\nОшибка при обработке изображения {os.path.basename(file_path)}: {str(e)}"
                    else:
                        content.append({"type": "text", "text": f"Ошибка при обработке изображения {os.path.basename(file_path)}: {str(e)}"})
    
    # Если есть текстовые файлы, добавляем их содержимое к сообщению
    if file_contents:
        if message and content:
            content[0]["text"] += "\n\n" + "\n\n".join(file_contents)
        else:
            content.append({"type": "text", "text": "\n\n".join(file_contents)})
    
    # Если контент пустой, добавляем пустое текстовое сообщение
    if not content:
        content.append({"type": "text", "text": ""})
    
    return content

# Глобальная переменная для отслеживания активных запросов
active_requests = {}

def process_message(message, history, files, model_name, temperature, max_tokens, system_message):
    # Создаем уникальный ID для запроса
    request_id = str(uuid.uuid4())
    active_requests[request_id] = True
    
    try:
        # Подготовка сообщений из истории
        messages = []
        
        # Добавляем системное сообщение, если оно есть
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        if history:
            for user_msg, assistant_msg in history:
                messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
        
        # Добавление текущего сообщения
        content = prepare_message_content(message, files)
        messages.append({"role": "user", "content": content})
        
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
        response = requests.post(
            "https://glama.ai/api/gateway/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            json=request_data,
            stream=True
        )
        
        response_text = ""
        for line in response.iter_lines():
            if not active_requests.get(request_id, False):
                # Запрос был отменен
                yield "[Генерация ответа отменена]"
                return
            
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk.get('choices') and chunk['choices'][0].get('delta') and chunk['choices'][0]['delta'].get('content'):
                            content = chunk['choices'][0]['delta']['content']
                            response_text += content
                            yield response_text
                    except json.JSONDecodeError:
                        pass
        
        return response_text
    except Exception as e:
        return f"Ошибка при обращении к API: {str(e)}"
    finally:
        # Удаляем запрос из активных
        if request_id in active_requests:
            del active_requests[request_id]

# Функция для отмены текущего запроса
def cancel_generation():
    for request_id in list(active_requests.keys()):
        active_requests[request_id] = False
    return "Генерация ответа отменена"

# Функция для очистки временных файлов
def cleanup_temp_files():
    try:
        for file in TEMP_DIR.glob("*"):
            # Удаляем файлы старше 1 часа
            if time.time() - file.stat().st_mtime > 3600:
                file.unlink()
    except Exception as e:
        print(f"Ошибка при очистке временных файлов: {str(e)}")

# Запускаем очистку временных файлов в отдельном потоке
def start_cleanup_thread():
    def cleanup_worker():
        while True:
            cleanup_temp_files()
            time.sleep(3600)  # Запускаем очистку каждый час
    
    thread = threading.Thread(target=cleanup_worker, daemon=True)
    thread.start()

# Получаем список моделей при запуске
available_models = get_available_models()
default_model = available_models[0] if available_models else "gpt-4.5-preview-2025-02-27"

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
    # Состояние приложения
    current_request_id = gr.State(None)
    
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
    
    file_upload = gr.File(file_count="multiple", label="Прикрепить файлы", file_types=["image", "text", ".pdf", ".doc", ".docx"])
    
    with gr.Row():
        clear = gr.Button("Очистить чат")
        cancel_btn = gr.Button("Отменить генерацию", variant="stop")
        export_btn = gr.Button("Экспорт истории")
    
    # Индикатор состояния
    status_indicator = gr.Markdown("Готов к работе")
    
    # Функция для обработки сообщений
    def user_input(message, chat_history, files, model_name, temperature, max_tokens, system_msg):
        # Проверка на None
        if chat_history is None:
            chat_history = []
            
        if not message and not files:
            return "", chat_history, None, "Введите сообщение или прикрепите файлы"
        
        # Обновляем статус
        yield "", chat_history, None, "Обработка сообщения..."
        
        # Добавляем сообщение пользователя в историю
        chat_history.append((message, None))
        yield "", chat_history, None, "Генерация ответа..."
        
        # Получаем ответ от модели
        bot_message = ""
        for response in process_message(message, chat_history[:-1], files, model_name, temperature, max_tokens, system_msg):
            bot_message = response
            chat_history[-1] = (message, bot_message)
            yield "", chat_history, None, "Генерация ответа..." if "[Генерация ответа отменена]" not in bot_message else "Генерация отменена"
        
        return "", chat_history, None, "Готов к работе"
    
    # Функция для обновления списка моделей
    def refresh_models():
        new_models = get_available_models(force_refresh=True)
        return gr.Dropdown(choices=new_models, value=new_models[0] if new_models else None), "Список моделей обновлен"
    
    # Функция для экспорта истории чата
    def export_history(chat_history):
        if chat_history is None or len(chat_history) == 0:
            return None, "История чата пуста"
        
        try:
            export_text = "# История чата\n\n"
            for user_msg, bot_msg in chat_history:
                export_text += f"## Пользователь:\n{user_msg}\n\n"
                if bot_msg:
                    export_text += f"## Ассистент:\n{bot_msg}\n\n"
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"chat_history_{timestamp}.md"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(export_text)
            
            return filename, "История чата экспортирована"
        except Exception as e:
            return None, f"Ошибка при экспорте: {str(e)}"
    
    # Функция для сохранения чата
    def save_chat(chat_history):
        if chat_history is None or len(chat_history) == 0:
            return None, "История чата пуста"
        
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"saved_chat_{timestamp}.json"
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(chat_history, f, ensure_ascii=False, indent=2)
            
            return filename, "Чат сохранен"
        except Exception as e:
            return None, f"Ошибка при сохранении: {str(e)}"
    
    # Функция для загрузки чата
    def load_chat(file):
        if not file:
            return [], "Файл не выбран"
        
        try:
            with open(file.name, "r", encoding="utf-8") as f:
                chat_history = json.load(f)
            
            return chat_history, "Чат загружен"
        except Exception as e:
            return [], f"Ошибка при загрузке: {str(e)}"
    
    # Функция для очистки чата
    def clear_chat():
        return [], None, "Чат очищен"
    
    # Привязка событий
    msg.submit(user_input, [msg, chatbot, file_upload, model_dropdown, temperature_slider, max_tokens_slider, system_message], 
               [msg, chatbot, file_upload, status_indicator])
    
    submit_btn.click(user_input, [msg, chatbot, file_upload, model_dropdown, temperature_slider, max_tokens_slider, system_message], 
                    [msg, chatbot, file_upload, status_indicator])
    
    clear.click(clear_chat, outputs=[chatbot, file_upload, status_indicator])
    
    cancel_btn.click(cancel_generation, outputs=[status_indicator])
    
    export_btn.click(export_history, inputs=[chatbot], outputs=[gr.File(label="Скачать историю"), status_indicator])
    
    refresh_models_btn.click(refresh_models, outputs=[model_dropdown, status_indicator])
    
    # Добавляем кнопки для сохранения и загрузки чата
    with gr.Row():
        save_chat_btn = gr.Button("Сохранить чат")
        load_chat_file = gr.File(label="Загрузить чат", file_types=[".json"])
    
    save_chat_btn.click(save_chat, inputs=[chatbot], outputs=[gr.File(label="Скачать сохраненный чат"), status_indicator])
    load_chat_file.change(load_chat, inputs=[load_chat_file], outputs=[chatbot, status_indicator])

# Запускаем очистку временных файлов
start_cleanup_thread()

# Запуск приложения
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
