from pydub import AudioSegment
from io import BytesIO, StringIO
import os
import pydub
from google.cloud import texttospeech
from copy import deepcopy


def create_client():
    return texttospeech.TextToSpeechClient()


__voices_table = {}
__LANGUAGES_TABLE = {
    'af-ZA': '南非荷蘭語-南非',
    'sq-AL': '阿爾巴尼亞人-阿爾巴尼亞',
    'ar-DZ': '阿拉伯語-阿爾及利亞',
    'ar-BH': '阿拉伯語-巴林',
    'ar-EG': '阿拉伯語-埃及',
    'ar-IQ': '阿拉伯語-伊拉克',
    'ar-JO': '阿拉伯語-約旦',
    'ar-KW': '阿拉伯語-科威特',
    'ar-LB': '阿拉伯語-黎巴嫩',
    'ar-LY': '阿拉伯語-利比亞',
    'ar-MA': '阿拉伯語-摩洛哥',
    'ar-OM': '阿拉伯語-阿曼',
    'ar-QA': '阿拉伯語-卡塔爾',
    'ar-SA': '阿拉伯語-沙特阿拉伯',
    'ar-SY': '阿拉伯語-敘利亞',
    'ar-TN': '阿拉伯語-突尼斯',
    'ar-AE': '阿拉伯語-阿拉伯聯合酋長國',
    'ar-YE': '阿拉伯語-也門',
    'hy-AM': '亞美尼亞-亞美尼亞',
    'Cy-az-AZ': '阿塞拜疆（西里爾）-阿塞拜疆',
    'Lt-az-AZ': '阿塞拜疆（拉丁）-阿塞拜疆',
    'eu-ES': '巴斯克語-巴斯克語',
    'be-BY': '白俄羅斯-白俄羅斯',
    'bg-BG': '保加利亞-保加利亞',
    'ca-ES': '加泰羅尼亞語-加泰羅尼語',
    'zh-CN': '中文-中國',
    'zh-HK': '中文-香港特別行政區',
    'zh-MO': '中文-澳門特別行政區',
    'zh-SG': '中文-新加坡',
    'zh-TW': '中文-台灣',
    'zh-CHS': '簡體中文）',
    'zh-CHT': '繁體中文）',
    'hr-HR': '克羅地亞-克羅地亞',
    'cs-CZ': '捷克-捷克共和國',
    'da-DK': '丹麥語-丹麥語',
    'div-MV': 'Dhivehi-馬爾地夫',
    'nl-BE': '荷蘭-比利時',
    'nl-NL': '荷蘭-荷蘭',
    'en-AU': '英文-澳大利亞',
    'en-BZ': '英語-伯利茲',
    'en-CA': '英文-加拿大',
    'en-CB': '英語-加勒比',
    'en-IE': '英語-愛爾蘭',
    'en-JM': '英語-牙買加',
    'en-NZ': '英語-新西蘭',
    'en-PH': '英語-菲律賓',
    'en-ZA': '英文-南非',
    'en-TT': '英文-特立尼達和多巴哥',
    'en-GB': '英文-英國',
    'en-US': '美國英語',
    'en-ZW': '英語-津巴布韋',
    'et-EE': '愛沙尼亞語-愛沙尼亞',
    'fo-FO': '法羅群島-法羅群島',
    'fa-IR': '波斯-伊朗',
    'fi-FI': '芬蘭-芬蘭',
    'fr-BE': '法語-比利時',
    'fr-CA': '法語-加拿大',
    'fr-FR': '法國-法國',
    'fr-LU': '法語-盧森堡',
    'fr-MC': '法語-摩納哥',
    'fr-CH': '法語-瑞士',
    'gl-ES': '加利西亞人-加利西亞人',
    'ka-GE': '格魯吉亞-格魯吉亞',
    'de-AT': '德國-奧地利',
    'de-DE': '德國-德國',
    'de-LI': '德語-列支敦士登',
    'de-LU': '德國-盧森堡',
    'de-CH': '德國-瑞士',
    'el-GR': '希臘-希臘',
    'gu-IN': '古吉拉特文-印度',
    'he-IL': '希伯來語-以色列',
    'hi-IN': '印地文-印度',
    'hu-HU': '匈牙利-匈牙利',
    'is-IS': '冰島-冰島',
    'id-ID': '印尼-印尼',
    'it-IT': '意大利-意大利',
    'it-CH': '意大利語-瑞士',
    'ja-JP': '日文-日本',
    'kn-IN': '卡納達-印度',
    'kk-KZ': '哈薩克斯坦',
    'kok-IN': '康卡尼-印度',
    'ko-KR': '韓國-韓國',
    'ky-KZ': '吉爾吉斯斯坦-哈薩克斯坦',
    'lv-LV': '拉脫維亞-拉脫維亞',
    'lt-LT': '立陶宛立陶宛',
    'mk-MK': '馬其頓（FYROM）',
    'ms-BN': '馬來文-文萊',
    'ms-MY': '馬來文-馬來西亞',
    'mr-IN': '馬拉地-印度',
    'mn-MN': '蒙古-蒙古',
    'nb-NO': '挪威語（Bokmål）-挪威',
    'nn-NO': '挪威（尼諾斯克）-挪威',
    'pl-PL': '波蘭語-波蘭語',
    'pt-BR': '葡萄牙語-巴西',
    'pt-PT': '葡萄牙-葡萄牙',
    'pa-IN': '旁遮普邦-印度',
    'ro-RO': '羅馬尼亞語-羅馬尼亞',
    'ru-RU': '俄羅斯-俄羅斯',
    'sa-IN': '梵文-印度',
    'Cy-sr-SP': '塞爾維亞語（塞爾維亞語）-塞爾維亞語',
    'Lt-sr-SP': '塞爾維亞（拉丁）-塞爾維亞',
    'sk-SK': '斯洛伐克語-斯洛伐克',
    'sl-SI': '斯洛文尼亞語-斯洛維尼亞',
    'es-AR': '西班牙語-阿根廷',
    'es-BO': '西班牙語-玻利維亞',
    'es-CL': '西班牙語-智利',
    'es-CO': '西班牙-哥倫比亞',
    'es-CR': '西班牙語-哥斯達黎加',
    'es-DO': '西班牙語-多米尼加共和國',
    'es-EC': '西班牙語-厄瓜多爾',
    'es-SV': '西班牙語-薩爾瓦多',
    'es-GT': '西班牙語-危地馬拉',
    'es-HN': '西班牙語-洪都拉斯',
    'es-MX': '西班牙語-墨西哥',
    'es-NI': '西班牙語-尼加拉瓜',
    'es-PA': '西班牙語-巴拿馬',
    'es-PY': '西班牙語-巴拉圭',
    'es-PE': '西班牙語-秘魯',
    'es-PR': '西班牙語-波多黎各',
    'es-ES': '西班牙語-西班牙',
    'es-UY': '西班牙語-烏拉圭',
    'es-VE': '西班牙語-委內瑞拉',
    'sw-KE': '斯瓦希里語-肯尼亞',
    'sv-FI': '瑞典語-芬蘭語',
    'sv-SE': '瑞典語-瑞典語',
    'syr-SY': '敘利亞-敘利亞',
    'ta-IN': '泰米爾語-印度',
    'tt-RU': '韃靼-俄羅斯',
    'te-IN': '泰盧固語-印度',
    'th-TH': '泰國-泰國',
    'tr-TR': '土耳其-土耳其',
    'uk-UA': '烏克蘭-烏克蘭',
    'ur-PK': '烏爾都語-巴基斯坦',
    'Cy-uz-UZ': '烏茲別克語（西里爾文）-烏茲別克斯坦',
    'Lt-uz-UZ': '烏茲別克語（拉丁語）-烏茲別克斯坦',
    'vi-VN': '越南-越南',
    'ar-XA': '阿拉伯語-其他亞洲國家',
    'bn-IN': '孟加拉語-印度',
    'ml-IN': 'ml-印度',
    'en-IN': '英語-印度',
    'es-US': '西班牙-美國',
    'cmn-CN': '普通話-中國',
    'cmn-TW': '普通話-台灣',
    'fil-PH': 'fil-菲律賓',
    'yue-HK': '粵語-香港',
    'sr-RS': '塞爾維亞-塞爾維亞',
}


def list_voices(client):
    if len(__voices_table) > 0:
        return deepcopy(__voices_table)
    voices = {}
    for i, v in enumerate(sorted(client.list_voices().voices, key=lambda v: v.name)):
        voices[v.name] = "{}-{}-{}".format(i, v.name, v.ssml_gender.name)

    for voice in client.list_voices().voices:
        for code in voice.language_codes:
            if code not in __voices_table:
                __voices_table[code] = []
            __voices_table[code].append(
                {'name': voice.name, 'display': voices[voice.name]})
    return deepcopy(__voices_table)


def list_languages():
    return __LANGUAGES_TABLE.copy()


def to_audio(client, text, lang, rate=1.0, voice="en-US-Wavenet-D"):
    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(ssml=text)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice_obj = texttospeech.VoiceSelectionParams(
        language_code=lang,
        name=voice,
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.OGG_OPUS,
        speaking_rate=rate,
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice_obj, audio_config=audio_config
    )

    # The response's audio_content is binary.
    buffer = BytesIO()
    buffer.write(response.audio_content)
    buffer.seek(0)
    seg = AudioSegment.from_file(buffer)
    return seg, seg.duration_seconds


def limit_texts(texts, limit=4500):
    new_texts = []
    for text in texts:
        text = text.strip()
        if len(text) == 0:
            continue
        if len(text) < limit:
            new_texts.append(text)
            continue
        for t in limit_texts(text.split('\n')):
            if len(t) < limit:
                new_texts.append(t)
                continue
            for tt in t.split('. '):
                new_texts.append(tt)
    return new_texts


def script_to_audio(client, src, lang='en-US', rate=0.9, voice="en-US-Wavenet-D"):
    if isinstance(src, str):
        if os.path.exists(src):
            thefile = open(src, 'r')
        else:
            thefile = StringIO(src)
    elif getattr(src, 'read', None) is not None:
        thefile = src
    raw = thefile.read().strip('---').strip()

    is_ssml = raw.startswith('<speak>') and raw.endswith('</speak>')

    texts = limit_texts(raw.split('---'))
    audios = []
    len_texts = len(texts)
    for i, text in enumerate(texts):
        if is_ssml and len_texts > 1:
            if i == 0:
                text += "</speak>"
            elif i == len_texts - 1:
                text = "<speak>" + text
            else:
                text = "<speak>" + text + "</speak>"
        a, seconds = to_audio(client, text, lang, rate=rate, voice=voice)
        audios.append(a)
        silent = pydub.AudioSegment.silent(duration=500)
        audios.append(silent)

    output = BytesIO()
    sum(audios).set_channels(2).export(output)
    return output
