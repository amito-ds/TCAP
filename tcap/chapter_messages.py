import random

CLEANING_EMOJIS = ["๐งน", "๐งผ", "๐งฝ", "๐งด"]
PREPROCESSING_EMOJIS = ["๐", "๐", "๐ฌ", "๐ง"]
TEXT_ANALYSIS_EMOJIS = ["๐", "๐", "๐", "๐"]
MODEL_PRE_ANALYSIS_EMOJIS = ["๐", "๐", "๐ฌ", "๐งฐ"]
MODEL_RUN_EMOJIS = ["๐ค", "๐ฅ", "๐ง ", "๐"]
POST_MODEL_ANALYSIS_EMOJIS = ["๐", "๐", "๐ฌ", "๐"]
CREATE_EMBEDDING_EMOJIS = ["๐", "๐ฌ", "๐ง ", "๐"]


def chapter_message(chapter_name: str, prefix=" Running chapter: "):
    stars = '*' * (len(chapter_name + prefix) + 12)
    if chapter_name == 'cleaning':
        emoji = random.choice(CLEANING_EMOJIS)
    elif chapter_name == 'preprocessing':
        emoji = random.choice(PREPROCESSING_EMOJIS)
    elif chapter_name == 'text analyze':
        emoji = random.choice(TEXT_ANALYSIS_EMOJIS)
    elif chapter_name == 'model pre analysis':
        emoji = random.choice(MODEL_PRE_ANALYSIS_EMOJIS)
    elif chapter_name == 'model training':
        emoji = random.choice(MODEL_RUN_EMOJIS)
    elif chapter_name == 'post model analysis':
        emoji = random.choice(POST_MODEL_ANALYSIS_EMOJIS)
    elif chapter_name == 'create embedding':
        emoji = random.choice(CREATE_EMBEDDING_EMOJIS)
    else:
        emoji = '๐ต'
    message = f'{stars}\n***{prefix} {chapter_name} {emoji} ***\n{stars}'
    return message



