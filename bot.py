import logging
import pickle
import nest_asyncio
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from preprosessing_file import preprossesing
from catboost import CatBoostClassifier

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = 'cat_boost_balanced.cbm'
catboost_model = CatBoostClassifier()
catboost_model.load_model(model_path)

dct_path = 'dct.pkl'
with open(dct_path, 'rb') as f:
    loaded_dct = pickle.load(f)

API_TOKEN = 'token'

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    raw_str = update.message.text
    raw_data = raw_str.split()
    try:
        processed = preprossesing(raw_data, loaded_dct)
        prediction = catboost_model.predict(processed)
        result = int(prediction[0])
        await update.message.reply_text(f"Предсказание: {result}")
    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        await update.message.reply_text("Не удалось выполнить предсказание. Проверьте формат входных данных.")

async def main() -> None:
    nest_asyncio.apply()
    application = Application.builder().token(API_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    await application.run_polling()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
