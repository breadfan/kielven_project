from telegram import (
    ReplyKeyboardMarkup, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup, 
    Update, 
    InputMediaPhoto
)
from telegram.ext import (
    ApplicationBuilder, CommandHandler, 
    CallbackQueryHandler, CallbackContext, 
    MessageHandler, filters, 
    ContextTypes, ConversationHandler, 
    AIORateLimiter
)
from telegram.constants import ParseMode
import pandas as pd
import numpy as np
import os
from tqdm.auto import trange
import html
import logging
import traceback
import yaml
import pickle
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, NllbTokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
from sklearn.metrics.pairwise import cosine_similarity
import requests


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

BOT_TOKEN = config['bot_token']
DATA_FOLDER = config['data_folder'] # 'data/'
DEVELOPER_CHAT_ID = config['developer_chat_id']
NLLB_PATH = 'weights/nllb/'
MBART_PATH = 'weights/mbart/'

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ID совершённой операции для conversational bot
CHOOSING, NEXT, ENTER_TEXT = range(3)
BUT1 = "Размечать"
BUT2 = "Информация о текущей разметке"
BUT3, BUT4 = "Перевод KAR-RU", "Перевод RU-KAR" 
BUT5 = "Помощь"

main_keyboard = [
    [BUT1],
    [BUT2],
    [BUT3, BUT4],
    [BUT5]
]
main_markup = ReplyKeyboardMarkup(main_keyboard, one_time_keyboard=True)

class LabellingBot:
    def __init__(self) -> None:
        self.lines = None
        self.last_row_id = None # id последней размеченной строки
        self.labeled = list() # размеченные данные
        self.data_path = DATA_FOLDER 
        self.stack = list() 
        self.nllb_model = None
        self.mbart_model = None
        self.nllb_tokenizer = None
        self.mbart_tokenizer = None
        self.dict_embeds = None
        self.kar_to_rus = None
        self.dict_keys = None
        self.cache = None
        self.get_cache()
        self.load_models()


    def get_cache(self):
        cache_path = os.path.join(DATA_FOLDER, 'trans_10k_all.csv')
        if os.path.isfile(cache_path):
            self.cache = pd.read_csv(cache_path)
    
    def load_models(self):  
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_PATH)
        self.nllb_tokenizer = NllbTokenizer.from_pretrained(NLLB_PATH)
        self.nllb_tokenizer.src_lang = 'olo_Latn' 
        self.nllb_tokenizer.tgt_lang = 'rus_Cyrl'
        self.mbart_model = MBartForConditionalGeneration.from_pretrained(MBART_PATH)
        self.mbart_tokenizer = MBart50TokenizerFast.from_pretrained(MBART_PATH)
        self.mbart_tokenizer.src_lang = 'fi_FI'
        self.mbart_tokenizer.tgt_lang = 'ru_RU'
        self.dict_embeds = np.load(os.path.join(self.data_path, 'dict_embeds.npy'))
        with open(os.path.join(self.data_path, 'saved_dictionary.pkl'), 'rb') as f:
            self.kar_to_rus = pickle.load(f)
        self.dict_keys = list(self.kar_to_rus.keys())
        
        
    # весь код для загрузки данных для перевода 
    async def load_corpus(self, update: Update, context: CallbackContext):
        # загрузка данных из файла для разметки предложений
        file_path = os.path.join(self.data_path, 'corpus_to_label.txt')
        if not os.path.isfile(file_path):
            await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Файл corpus_to_label.txt не найден. Пожалуйста, загрузите файл в папку пользователя."
            )
            return
        
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        if not lines:
            await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Файл corpus_to_label.txt пуст. Пожалуйста, проверьте содержимое файла."
            )
            return
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Файл corpus_to_label.txt загружен. Начинаем разметку."
            )
        self.lines = lines
        context.user_data['labelling_id'] = 0

    def get_embed(self, text, model, tokenizer, src_lang, tgt_lang, normalize_embeddings=True):
        t = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            res = model.generate(**t, 
                                return_dict_in_generate=True, 
                                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                                output_hidden_states=True)
            per_token_embeddings = res['encoder_hidden_states'][-1]
            mask = t.attention_mask
            embeddings = (per_token_embeddings * mask.unsqueeze(-1)).sum(1) / mask.unsqueeze(-1).sum(1)
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings.squeeze(-1).cpu().numpy()

    def batched_embed(self, texts, batch_size=16, **kwargs):
        """Translate texts in batches of similar length"""
        idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
        results = []
        for i in trange(0, len(texts2), batch_size):
            results.extend(self.get_embed(texts2[i: i+batch_size], **kwargs))
        return np.array([p for i, p in sorted(zip(idxs, results))])
    
    def translate(
        self, 
        text, 
        model, tokenizer,
        src_lang, tgt_lang, 
        a=32, b=3, max_input_length=1024, num_beams=4, 
        **kwargs
    ):
        """Turn a text or a list of texts into a list of translations"""
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
        inputs = tokenizer(
            text, return_tensors='pt', padding=True, truncation=True, 
            max_length=max_input_length
        )
        model.eval() # turn off training mode
        result = model.generate(
            **inputs.to(model.device),
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
            num_beams=num_beams, **kwargs
        )
        return tokenizer.batch_decode(result, skip_special_tokens=True)

    def batched_translate(self, texts, batch_size=16, **kwargs):
        """Translate texts in batches of similar length"""
        idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
        results = []
        for i in trange(0, len(texts2), batch_size):
            results.extend(self.translate(texts2[i: i+batch_size], **kwargs))
        return [p for i, p in sorted(zip(idxs, results))]
    
    def get_dict_translate(self, text):
        text_split = text.split() 
        if len(text_split) > 1:
            embeds = self.batched_embed(text_split, 
                                        model=self.nllb_model, 
                                        tokenizer=self.nllb_tokenizer, 
                                        src_lang='olo_Latn', 
                                        tgt_lang='rus_Cyrl')
        else:
            embeds = self.get_embed(text_split,
                                    model=self.nllb_model, 
                                    tokenizer=self.nllb_tokenizer, 
                                    src_lang='olo_Latn', 
                                    tgt_lang='rus_Cyrl')
        if embeds.ndim == 1: # если одномерный массив
            embeds = embeds.reshape(1,-1)
        ids_closest = cosine_similarity(embeds, self.dict_embeds).argmax(axis=1)
        kar_words = [self.kar_to_rus[self.dict_keys[id_closest]] for id_closest in ids_closest]
        pairs = list(zip(text_split, kar_words))
        return pairs

    def get_translates(self, text, src_rus=False):
        if not src_rus:
            nllb_src, nllb_tgt = 'olo_Latn', 'rus_Cyrl'
            mbart_src, mbart_tgt = 'fi_FI', 'ru_RU'
        else:
            nllb_src, nllb_tgt = 'rus_Cyrl', 'olo_Latn'
            mbart_src, mbart_tgt = 'ru_RU', 'fi_FI'
        nllb_trans = self.translate(text, self.nllb_model, self.nllb_tokenizer, 
                                    src_lang=nllb_src, tgt_lang=nllb_tgt)
        mbart_trans = self.translate(text, self.mbart_model, self.mbart_tokenizer, 
                                    src_lang=mbart_src, tgt_lang=mbart_tgt)
        pairs = self.get_dict_translate(text)
        return nllb_trans, mbart_trans, pairs
    
    
    async def start(self, update: Update, context: CallbackContext):
        context.user_data['labelling'] = True
        await self.load_corpus(update, context)
        return await self.send_next_pair(update, context)
        
    async def save_labels(self, update:Update, context: CallbackContext) -> None:
        if len(self.labeled) > 0:
            # если уже есть разметка для данного юзера, то надо добавить текущую к имеющейся
            df = pd.DataFrame(self.labeled, columns=['kar', 'rus'])
            df.dropna(inplace=True, ignore_index=True)
            # заполняем -1 NaN, чтобы затем их отличить     
            # self.df.fillna(-1, inplace=True)
            labeled_path = os.path.join(self.data_path, 'labeled.csv')
            if os.path.isfile(labeled_path):
                old = pd.read_csv(labeled_path)
                df = pd.concat([old, df], ignore_index=True)
            df.drop_duplicates(ignore_index=True, inplace=True)
            df.to_csv(labeled_path, index=False)
            await context.bot.send_message(chat_id=update.effective_chat.id, text='Разметка сохранена!')
            labelling_id = context.user_data.get('labelling_id', 0)
            if self.cache is not None:
                self.cache.to_csv(os.path.join(DATA_FOLDER, 'trans_10k_all.csv'), index=False)
            # Save the remaining lines to a file
            with open(os.path.join(DATA_FOLDER, 'corpus_to_label.txt'), 'w', encoding='utf-8') as file:
                file.writelines(self.lines[labelling_id:])
            await context.bot.send_message(chat_id=update.effective_chat.id, text='Оставшиеся строки сохранены в remaining_corpus.txt.')
        # else:
        #     await context.bot.send_message(chat_id=update.effective_chat.id,text='Пустая таблица/не было разметки. Сохранения не произошло')

    async def show_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_data = context.user_data
        if 'article_name' in user_data:
            msg = f"Название статьи: {user_data['article_name']}\n"
        else:
            msg = "Статья не указана\n"
        await update.message.reply_text(
            msg,
            reply_markup=main_markup,
        )
    
    async def stop(self, update: Update, context: CallbackContext) -> None:
        """ Прекращение работы алгоритма. Сохранение разметки и недоразмеченных данных. """
        # очищаем условия
        context.user_data['labelling'] = False
        await self.save_labels(update, context)
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text='Бот остановлен, спасибо!\nЧтобы начать заново, нажмите /start'
        )
        return ConversationHandler.END

    def get_tartu_trans(self, text, src='olo', tgt='rus'):
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
        }
        json_data = {
            'text': text,
            'src': src,
            'tgt': tgt,
            'domain': 'general',
            'application': 'Documentation UI',
        }
        try:
            response = requests.post('https://api.tartunlp.ai/translation/v2', headers=headers, json=json_data)
            res = response.json()['result']
        except:
            res = ''
        return res
    
    async def send_next_pair(self, update: Update, context: CallbackContext):
        if not context.user_data['labelling']:
            return
        elif 'skip' in context.user_data:
            del context.user_data['skip']
        elif 'translate' in context.user_data:
            translate = update.message.text
            self.labeled.append((self.orig_sentence, translate))
            # await update.message.reply_text(f"Оригинал: {self.orig_sentence}")
            # await update.message.reply_text(f"Вы ответили: {translate}")
        labelling_id = context.user_data['labelling_id'] 
        line = self.lines[labelling_id].strip()
        line_split = line.split('.')
        if len(line_split) < 2 and line[-1] != '.':
            labelling_id += 1
            context.user_data['article_name'] = line_split[0]
            line = self.lines[labelling_id].strip()
            line_split = line.split('.')
        self.orig_sentence = line_split[0]
        await context.bot.send_message(chat_id=update.effective_chat.id, text=self.orig_sentence)
        if self.cache is not None:
            nllb, mbart, pairs, _, tartu = self.cache.iloc[labelling_id]
            pairs = pairs[1:-1].split('), ')
            pairs = [tuple(p[1:].split(', \'')) for p in pairs]
            pairs = [(f[1:-1], s[:-1]) for f, s in pairs]
            combined_translation = f"NLLB:\n{nllb}\n\nMBART:\n{mbart}\n\nTARTU:\n{tartu}\n\nDictionary:\n{'\n'.join([f'{k} -> {v}' for k, v in pairs])}"
        else:
            nllb, mbart, pairs = self.get_translates(self.orig_sentence)
            tartu = self.get_tartu_trans(self.orig_sentence)
            combined_translation = f"NLLB:\n{nllb[0]}\n\nMBART:\n{mbart[0]}\n\nTARTU:\n{tartu}\n\nDictionary:\n{'\n'.join([f'{k} -> {v}' for k, v in pairs])}"
        await context.bot.send_message(chat_id=update.effective_chat.id, text=combined_translation)
        labelling_id += 1
        context.user_data['labelling_id'] = labelling_id
        # сохранение каждые 20 предложений
        if context.user_data['labelling_id'] % 20 == 0: 
            await self.save_labels(update, context)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Сохранение разметки")
        reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton("Пропустить", callback_data="skip")],
                                            [InlineKeyboardButton("Легенда", callback_data='legend')],
                                            [InlineKeyboardButton("Стоп", callback_data='stop')]
            ])
        await context.bot.send_message(chat_id=update.effective_chat.id, 
            text="Вы можете пропустить текущее предложение, нажав кнопку ниже.",
            reply_markup=reply_markup
        )
        if update.message is not None:
            text = update.message.text.lower()
            context.user_data['translate'] = text
        # TODO сохран
        return NEXT
        
    async def button(self, update: Update, context: CallbackContext):
        query = update.callback_query
        await query.answer()
        if query.data == 'skip':
            context.user_data['skip'] = True
            await self.send_next_pair(update, context)
        elif query.data == 'stop':
            await self.stop(update, context)
        elif query.data == 'legend':
            legend_txt = 'Первое сообщение — оригинал, второе — перевод NLLB, третье — перевод MBART, четвёртое — перевод TARTU '
            legend_txt += ' и пятое — перевод по словарю.\n'
            legend_txt += 'Если вы хотите пропустить строку, нажмите кнопку «Пропустить».\n'
            legend_txt += 'Если вы хотите остановить бота, нажмите кнопку «Стоп».\n'
            await context.bot.send_message(chat_id=update.effective_chat.id, text=legend_txt)
            return


    async def simple_translate(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        text = update.message.text.lower() # получает текст с клавиатуры либо написанный
        # вручную 'RU-KAR' или 'KAR-RU'
        if text.lower() == 'перевод ru-kar':
            context.user_data['source'] = 'ru'
        elif text.lower() == 'перевод kar-ru':
            context.user_data['source'] = 'kar'
        await update.message.reply_text(f"Введите предложение")
        return ENTER_TEXT
    
    async def enter_text(self, update: Update, context: CallbackContext) -> int:
        text = update.message.text
        source = context.user_data['source']
        if source == 'ru':
            nllb, mbart, _ = self.get_translates(text, src_rus=True)
            tartu = self.get_tartu_trans(text, src='rus', tgt='olo')
        else:
            nllb, mbart, _ = self.get_translates(text)
            tartu = self.get_tartu_trans(text)
        combined_translation = f"NLLB:\n{nllb[0]}\n\nMBART:\n{mbart[0]}\n\nTARTU:\n{tartu}\n\n"
        await context.bot.send_message(chat_id=update.effective_chat.id, text=combined_translation)
        await update.message.reply_text(f"Введите другое предложение или нажмите /stop, чтобы завершить перевод.")
        return ENTER_TEXT
        
    async def custom_text(self, update, context):
        if 'article_name' in context.user_data:
            article_name = context.user_data['article_name']
            await update.message.reply_text(f"Последний заголовок статьи:\n{article_name}")
        return NEXT

    async def ask_first_action(self, update: Update, context: CallbackContext) -> int:
        await update.message.reply_text(
            "Привет, это бот разметки. Выберите действие на клавиатуре",
            reply_markup=main_markup,
        )
        return CHOOSING

    
    async def help(self, update: Update, context: CallbackContext):
        help_query = 'Это бот разметки. Используется для пополнения параллельного корпуса для переводчика.'
        help_query += '\nНеобходимо \"Размечать\": начнётся поток сообщений, в котором вам будет предложено несколько версий перевода:\n'
        help_query += 'nllb, mbart и перевод со словаря.'
        help_query += '\nТакже доступен функционал обычного переводчика, для этого необходимо выбрать \"Перевод RU-KAR\" или \"Перевод KAR-RU\".'
        help_query += '\nДоступные команды:'
        help_query += '\n/start — начало взаимодействия с ботом'
        help_query += '\n/help — справка по боту'
        help_query += '\n/stop — конец взаимодействия с ботом; сохранение разметки'
        await update.message.reply_text(help_query, reply_markup=main_markup)
        return CHOOSING
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log the error and send a telegram message to notify the developer."""
        # Log the error before we do anything else, so we can see it even if something breaks.
        logger.error("Exception while handling an update:", exc_info=context.error)

        # traceback.format_exception returns the usual python message about an exception, but as a
        # list of strings rather than a single string, so we have to join them together.
        tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
        tb_string = "".join(tb_list)

        # Build the message with some markup and additional information about what happened.
        # You might need to add some logic to deal with messages longer than the 4096 character limit.
        update_str = update.to_dict() if isinstance(update, Update) else str(update)
        message = (
            "An exception was raised while handling an update\n"
            f"<pre>{html.escape(tb_string)}</pre>"
        )
        await self.save_labels(update, context)
        # отправляем сообщение разработчику об ошибке
        if len(message) < 4096:
            await context.bot.send_message(
                chat_id=DEVELOPER_CHAT_ID, text=message, parse_mode=ParseMode.HTML
            )
        else:
            await context.bot.send_message(
                chat_id=DEVELOPER_CHAT_ID, text='Строка ошибки длиннее, чем 4096.'
            )
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text="Произошла ошибка. Нажмите, пожалуйста, на /start" )
        
    def run(self) -> None:
        conv_handler = ConversationHandler(
            allow_reentry = True, 
            entry_points=[
                CommandHandler('start', self.ask_first_action),
                CommandHandler('text', self.custom_text),
                CommandHandler('help', self.help),
            ],
            states={
                CHOOSING: [
                    MessageHandler(filters.Regex(f"^{BUT1}|{BUT1.lower()}$"), self.start),
                    MessageHandler(filters.Regex(f"^{BUT2}$"), self.show_info),
                    MessageHandler(filters.Regex(f"^({BUT3}|{BUT3.lower()}|{BUT4}|{BUT4.lower()})$"), self.simple_translate),
                    MessageHandler(filters.Regex(f"^{BUT5}|{BUT5.lower()}$"), self.help),
                ],
                ENTER_TEXT: [
                    MessageHandler(filters.TEXT & ~(filters.COMMAND | filters.Regex("^Стоп|стоп$")), self.enter_text)
                ], 
                NEXT: [
                    MessageHandler(
                        filters.TEXT & ~(filters.COMMAND | filters.Regex("^Стоп|стоп$")), self.send_next_pair
                    )
                ],
            },
            fallbacks=[CommandHandler("stop", self.stop),
                       MessageHandler(filters.Regex("^Стоп|стоп$"), self.stop)],
        )

        application = ApplicationBuilder().token(BOT_TOKEN).read_timeout(7).\
                get_updates_read_timeout(42).rate_limiter(AIORateLimiter()).build()
        application.add_handler(CallbackQueryHandler(self.button))
        application.add_handler(conv_handler)
        application.add_error_handler(self.error_handler)

        application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    bot = LabellingBot()
    bot.run()