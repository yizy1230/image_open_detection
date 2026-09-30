"""Open vocabulary for subject + environment detection.

The previous copy from WeDetect-Anything was RAM's 4585-tag *tagging* list
(``ram/data/ram_tag_list_chinese.txt``). That list is a poor detector prompt:
it mixes actions, adjectives, abstract nouns, and scene labels that cannot be
localized. LLMDet and SAM 3 both consume **English noun phrases**.

This module is built from each model's training sources and what they can
actually localize:

* **LLMDet** (Grounding DINO + GroundingCap-1M)
  Objects365, GoldG (VG / Flickr30k / GQA), GRIT, V3Det, COCO, LLaVA-ReCap.
  Best at countable things with boxes. Prompt: ``"person . car . tree ."``.
  BERT budget ~256 tokens; official demos chunk ~40–80 names.

* **SAM 3** (promptable concept segmentation)
  ~4M noun phrases; SA-Co covers 270K concepts (MetaCLIP / SA-1B NPs,
  attributes, crowded scenes, wiki common / food / sports; silver: iNat,
  BDD100K, GeoDE, …). Best at stuff, parts, and short phrases. Prompt one
  concept at a time via ``set_text_prompt``.

Default export ``prompt_free_name`` is the **core** English list: common
foreground subjects + basic environment / stuff. Use ``get_vocab`` for
extended Objects365-style coverage or SAM 3 parts.
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional, Sequence

Role = Literal["subject", "environment", "part"]
Tier = Literal["core", "extended"]
Model = Literal["llmdet", "sam3"]

# (english, chinese, role, tier, models)
# models: ("llmdet", "sam3") unless SAM3-only (parts / amorphous stuff).
_ENTRY = tuple[str, str, Role, Tier, tuple[str, ...]]


def _both() -> tuple[str, ...]:
    return ("llmdet", "sam3")


def _sam3() -> tuple[str, ...]:
    return ("sam3",)


# ---------------------------------------------------------------------------
# Core subjects — COCO 80 + common people / animals / vehicles
# ---------------------------------------------------------------------------
_CORE_SUBJECT: list[_ENTRY] = [
    # people
    ("person", "人", "subject", "core", _both()),
    ("man", "男人", "subject", "core", _both()),
    ("woman", "女人", "subject", "core", _both()),
    ("boy", "男孩", "subject", "core", _both()),
    ("girl", "女孩", "subject", "core", _both()),
    ("child", "小孩", "subject", "core", _both()),
    ("baby", "婴儿", "subject", "core", _both()),
    ("crowd", "人群", "subject", "core", _both()),
    # animals (COCO + frequent O365)
    ("dog", "狗", "subject", "core", _both()),
    ("cat", "猫", "subject", "core", _both()),
    ("bird", "鸟", "subject", "core", _both()),
    ("horse", "马", "subject", "core", _both()),
    ("cow", "牛", "subject", "core", _both()),
    ("sheep", "羊", "subject", "core", _both()),
    ("bear", "熊", "subject", "core", _both()),
    ("elephant", "大象", "subject", "core", _both()),
    ("zebra", "斑马", "subject", "core", _both()),
    ("giraffe", "长颈鹿", "subject", "core", _both()),
    ("fish", "鱼", "subject", "core", _both()),
    ("rabbit", "兔子", "subject", "core", _both()),
    ("deer", "鹿", "subject", "core", _both()),
    ("tiger", "老虎", "subject", "core", _both()),
    ("lion", "狮子", "subject", "core", _both()),
    ("monkey", "猴子", "subject", "core", _both()),
    ("duck", "鸭子", "subject", "core", _both()),
    ("chicken", "鸡", "subject", "core", _both()),
    ("pig", "猪", "subject", "core", _both()),
    ("fox", "狐狸", "subject", "core", _both()),
    ("squirrel", "松鼠", "subject", "core", _both()),
    ("butterfly", "蝴蝶", "subject", "core", _both()),
    # vehicles
    ("car", "汽车", "subject", "core", _both()),
    ("bicycle", "自行车", "subject", "core", _both()),
    ("motorcycle", "摩托车", "subject", "core", _both()),
    ("bus", "公交车", "subject", "core", _both()),
    ("truck", "卡车", "subject", "core", _both()),
    ("train", "火车", "subject", "core", _both()),
    ("airplane", "飞机", "subject", "core", _both()),
    ("boat", "船", "subject", "core", _both()),
    ("ship", "轮船", "subject", "core", _both()),
    ("scooter", "滑板车", "subject", "core", _both()),
    # indoor / handheld objects
    ("chair", "椅子", "subject", "core", _both()),
    ("table", "桌子", "subject", "core", _both()),
    ("dining table", "餐桌", "subject", "core", _both()),
    ("sofa", "沙发", "subject", "core", _both()),
    ("bed", "床", "subject", "core", _both()),
    ("bench", "长凳", "subject", "core", _both()),
    ("bottle", "瓶子", "subject", "core", _both()),
    ("cup", "杯子", "subject", "core", _both()),
    ("bowl", "碗", "subject", "core", _both()),
    ("plate", "盘子", "subject", "core", _both()),
    ("book", "书", "subject", "core", _both()),
    ("cell phone", "手机", "subject", "core", _both()),
    ("laptop", "笔记本电脑", "subject", "core", _both()),
    ("computer", "电脑", "subject", "core", _both()),
    ("keyboard", "键盘", "subject", "core", _both()),
    ("mouse", "鼠标", "subject", "core", _both()),
    ("tv", "电视", "subject", "core", _both()),
    ("backpack", "背包", "subject", "core", _both()),
    ("handbag", "手提包", "subject", "core", _both()),
    ("umbrella", "伞", "subject", "core", _both()),
    ("clock", "时钟", "subject", "core", _both()),
    ("lamp", "灯", "subject", "core", _both()),
    ("vase", "花瓶", "subject", "core", _both()),
    ("potted plant", "盆栽", "subject", "core", _both()),
    # food
    ("apple", "苹果", "subject", "core", _both()),
    ("banana", "香蕉", "subject", "core", _both()),
    ("orange", "橙子", "subject", "core", _both()),
    ("pizza", "披萨", "subject", "core", _both()),
    ("cake", "蛋糕", "subject", "core", _both()),
    ("bread", "面包", "subject", "core", _both()),
    ("fruit", "水果", "subject", "core", _both()),
    ("vegetable", "蔬菜", "subject", "core", _both()),
    # sports / toys
    ("ball", "球", "subject", "core", _both()),
    ("skateboard", "滑板", "subject", "core", _both()),
    ("surfboard", "冲浪板", "subject", "core", _both()),
    ("teddy bear", "泰迪熊", "subject", "core", _both()),
]

# ---------------------------------------------------------------------------
# Core environment — COCO panoptic stuff + ADE20K-style scene elements
# ---------------------------------------------------------------------------
_CORE_ENVIRONMENT: list[_ENTRY] = [
    # sky / celestial
    ("sky", "天空", "environment", "core", _both()),
    ("cloud", "云", "environment", "core", _both()),
    ("sun", "太阳", "environment", "core", _both()),
    ("moon", "月亮", "environment", "core", _both()),
    # ground / terrain
    ("grass", "草地", "environment", "core", _both()),
    ("ground", "地面", "environment", "core", _both()),
    ("dirt", "泥土", "environment", "core", _both()),
    ("sand", "沙子", "environment", "core", _both()),
    ("rock", "岩石", "environment", "core", _both()),
    ("gravel", "碎石", "environment", "core", _both()),
    ("road", "道路", "environment", "core", _both()),
    ("sidewalk", "人行道", "environment", "core", _both()),
    ("path", "小路", "environment", "core", _both()),
    ("field", "田野", "environment", "core", _both()),
    ("hill", "山丘", "environment", "core", _both()),
    ("mountain", "山", "environment", "core", _both()),
    ("beach", "海滩", "environment", "core", _both()),
    ("desert", "沙漠", "environment", "core", _both()),
    # water
    ("water", "水", "environment", "core", _both()),
    ("sea", "海", "environment", "core", _both()),
    ("ocean", "海洋", "environment", "core", _both()),
    ("lake", "湖", "environment", "core", _both()),
    ("river", "河", "environment", "core", _both()),
    ("pond", "池塘", "environment", "core", _both()),
    ("waterfall", "瀑布", "environment", "core", _both()),
    ("wave", "波浪", "environment", "core", _both()),
    ("swimming pool", "游泳池", "environment", "core", _both()),
    ("fountain", "喷泉", "environment", "core", _both()),
    # vegetation
    ("tree", "树", "environment", "core", _both()),
    ("forest", "森林", "environment", "core", _both()),
    ("bush", "灌木", "environment", "core", _both()),
    ("flower", "花", "environment", "core", _both()),
    ("plant", "植物", "environment", "core", _both()),
    ("leaf", "叶子", "environment", "core", _both()),
    ("palm tree", "棕榈树", "environment", "core", _both()),
    # built outdoor
    ("building", "建筑", "environment", "core", _both()),
    ("house", "房子", "environment", "core", _both()),
    ("wall", "墙", "environment", "core", _both()),
    ("roof", "屋顶", "environment", "core", _both()),
    ("window", "窗户", "environment", "core", _both()),
    ("door", "门", "environment", "core", _both()),
    ("fence", "围栏", "environment", "core", _both()),
    ("bridge", "桥", "environment", "core", _both()),
    ("tower", "塔", "environment", "core", _both()),
    ("street", "街道", "environment", "core", _both()),
    # indoor setting
    ("floor", "地板", "environment", "core", _both()),
    ("ceiling", "天花板", "environment", "core", _both()),
    ("stairs", "楼梯", "environment", "core", _both()),
    ("carpet", "地毯", "environment", "core", _both()),
    ("curtain", "窗帘", "environment", "core", _both()),
    # weather / atmosphere (localizable enough for both)
    ("snow", "雪", "environment", "core", _both()),
    ("rain", "雨", "environment", "core", _both()),
    ("fog", "雾", "environment", "core", _sam3()),
    ("fire", "火", "environment", "core", _both()),
    ("smoke", "烟", "environment", "core", _both()),
]

# ---------------------------------------------------------------------------
# Extended subjects — frequent Objects365 / Open Images / LVIS things
# ---------------------------------------------------------------------------
_EXTENDED_SUBJECT: list[_ENTRY] = [
    # more people-adjacent
    ("rider", "骑手", "subject", "extended", _both()),
    ("worker", "工人", "subject", "extended", _both()),
    ("pedestrian", "行人", "subject", "extended", _both()),
    # more animals
    ("camel", "骆驼", "subject", "extended", _both()),
    ("penguin", "企鹅", "subject", "extended", _both()),
    ("dolphin", "海豚", "subject", "extended", _both()),
    ("whale", "鲸鱼", "subject", "extended", _both()),
    ("shark", "鲨鱼", "subject", "extended", _both()),
    ("parrot", "鹦鹉", "subject", "extended", _both()),
    ("swan", "天鹅", "subject", "extended", _both()),
    ("goose", "鹅", "subject", "extended", _both()),
    ("eagle", "鹰", "subject", "extended", _both()),
    ("pigeon", "鸽子", "subject", "extended", _both()),
    ("owl", "猫头鹰", "subject", "extended", _both()),
    ("goldfish", "金鱼", "subject", "extended", _both()),
    ("jellyfish", "水母", "subject", "extended", _both()),
    ("crab", "螃蟹", "subject", "extended", _both()),
    ("shrimp", "虾", "subject", "extended", _both()),
    ("lobster", "龙虾", "subject", "extended", _both()),
    ("turtle", "乌龟", "subject", "extended", _both()),
    ("snake", "蛇", "subject", "extended", _both()),
    ("lizard", "蜥蜴", "subject", "extended", _both()),
    ("frog", "青蛙", "subject", "extended", _both()),
    ("kangaroo", "袋鼠", "subject", "extended", _both()),
    ("panda", "熊猫", "subject", "extended", _both()),
    ("raccoon", "浣熊", "subject", "extended", _both()),
    ("wolf", "狼", "subject", "extended", _both()),
    ("goat", "山羊", "subject", "extended", _both()),
    ("donkey", "驴", "subject", "extended", _both()),
    ("llama", "羊驼", "subject", "extended", _both()),
    ("antelope", "羚羊", "subject", "extended", _both()),
    ("seal", "海豹", "subject", "extended", _both()),
    ("hamster", "仓鼠", "subject", "extended", _both()),
    ("peacock", "孔雀", "subject", "extended", _both()),
    ("bee", "蜜蜂", "subject", "extended", _both()),
    ("spider", "蜘蛛", "subject", "extended", _both()),
    # more vehicles
    ("van", "面包车", "subject", "extended", _both()),
    ("suv", "SUV", "subject", "extended", _both()),
    ("taxi", "出租车", "subject", "extended", _both()),
    ("helicopter", "直升机", "subject", "extended", _both()),
    ("ambulance", "救护车", "subject", "extended", _both()),
    ("fire truck", "消防车", "subject", "extended", _both()),
    ("tractor", "拖拉机", "subject", "extended", _both()),
    ("subway", "地铁", "subject", "extended", _both()),
    ("carriage", "马车", "subject", "extended", _both()),
    ("wheelchair", "轮椅", "subject", "extended", _both()),
    ("stroller", "婴儿车", "subject", "extended", _both()),
    ("sailboat", "帆船", "subject", "extended", _both()),
    # clothing / accessories (O365 frequent)
    ("hat", "帽子", "subject", "extended", _both()),
    ("glasses", "眼镜", "subject", "extended", _both()),
    ("sunglasses", "太阳镜", "subject", "extended", _both()),
    ("shoes", "鞋", "subject", "extended", _both()),
    ("boots", "靴子", "subject", "extended", _both()),
    ("coat", "外套", "subject", "extended", _both()),
    ("shirt", "衬衫", "subject", "extended", _both()),
    ("dress", "连衣裙", "subject", "extended", _both()),
    ("tie", "领带", "subject", "extended", _both()),
    ("helmet", "头盔", "subject", "extended", _both()),
    ("watch", "手表", "subject", "extended", _both()),
    ("necklace", "项链", "subject", "extended", _both()),
    # furniture / appliances
    ("cabinet", "橱柜", "subject", "extended", _both()),
    ("shelf", "架子", "subject", "extended", _both()),
    ("desk", "书桌", "subject", "extended", _both()),
    ("stool", "凳子", "subject", "extended", _both()),
    ("mirror", "镜子", "subject", "extended", _both()),
    ("pillow", "枕头", "subject", "extended", _both()),
    ("blanket", "毯子", "subject", "extended", _both()),
    ("towel", "毛巾", "subject", "extended", _both()),
    ("basket", "篮子", "subject", "extended", _both()),
    ("box", "盒子", "subject", "extended", _both()),
    ("trash can", "垃圾桶", "subject", "extended", _both()),
    ("sink", "水槽", "subject", "extended", _both()),
    ("toilet", "马桶", "subject", "extended", _both()),
    ("bathtub", "浴缸", "subject", "extended", _both()),
    ("refrigerator", "冰箱", "subject", "extended", _both()),
    ("oven", "烤箱", "subject", "extended", _both()),
    ("microwave", "微波炉", "subject", "extended", _both()),
    ("washing machine", "洗衣机", "subject", "extended", _both()),
    ("faucet", "水龙头", "subject", "extended", _both()),
    # electronics / tools
    ("camera", "相机", "subject", "extended", _both()),
    ("microphone", "麦克风", "subject", "extended", _both()),
    ("speaker", "音箱", "subject", "extended", _both()),
    ("remote", "遥控器", "subject", "extended", _both()),
    ("guitar", "吉他", "subject", "extended", _both()),
    ("piano", "钢琴", "subject", "extended", _both()),
    ("flag", "旗帜", "subject", "extended", _both()),
    ("toy", "玩具", "subject", "extended", _both()),
    ("suitcase", "行李箱", "subject", "extended", _both()),
    ("kite", "风筝", "subject", "extended", _both()),
    ("tennis racket", "网球拍", "subject", "extended", _both()),
    ("baseball bat", "棒球棒", "subject", "extended", _both()),
    ("scissors", "剪刀", "subject", "extended", _both()),
    ("knife", "刀", "subject", "extended", _both()),
    ("fork", "叉子", "subject", "extended", _both()),
    ("spoon", "勺子", "subject", "extended", _both()),
    ("wine glass", "酒杯", "subject", "extended", _both()),
    # more food
    ("tomato", "番茄", "subject", "extended", _both()),
    ("carrot", "胡萝卜", "subject", "extended", _both()),
    ("strawberry", "草莓", "subject", "extended", _both()),
    ("grape", "葡萄", "subject", "extended", _both()),
    ("watermelon", "西瓜", "subject", "extended", _both()),
    ("lemon", "柠檬", "subject", "extended", _both()),
    ("sandwich", "三明治", "subject", "extended", _both()),
    ("hamburger", "汉堡", "subject", "extended", _both()),
    ("hot dog", "热狗", "subject", "extended", _both()),
    ("donut", "甜甜圈", "subject", "extended", _both()),
    ("ice cream", "冰淇淋", "subject", "extended", _both()),
    ("sushi", "寿司", "subject", "extended", _both()),
    ("rice", "米饭", "subject", "extended", _both()),
    ("egg", "鸡蛋", "subject", "extended", _both()),
    ("cheese", "奶酪", "subject", "extended", _both()),
    ("coffee", "咖啡", "subject", "extended", _both()),
    ("wine", "葡萄酒", "subject", "extended", _both()),
    ("beer", "啤酒", "subject", "extended", _both()),
]

# ---------------------------------------------------------------------------
# Extended environment — T2I-useful scenes + ADE20K / BDD / GeoDE extras
# ---------------------------------------------------------------------------
_EXTENDED_ENVIRONMENT: list[_ENTRY] = [
    # sky / time
    ("star", "星星", "environment", "extended", _both()),
    ("rainbow", "彩虹", "environment", "extended", _both()),
    ("sunset", "日落", "environment", "extended", _sam3()),
    ("sunrise", "日出", "environment", "extended", _sam3()),
    ("night sky", "夜空", "environment", "extended", _both()),
    ("lightning", "闪电", "environment", "extended", _both()),
    # more terrain / nature
    ("valley", "山谷", "environment", "extended", _both()),
    ("cliff", "悬崖", "environment", "extended", _both()),
    ("canyon", "峡谷", "environment", "extended", _both()),
    ("cave", "洞穴", "environment", "extended", _both()),
    ("island", "岛", "environment", "extended", _both()),
    ("glacier", "冰川", "environment", "extended", _both()),
    ("ice", "冰", "environment", "extended", _both()),
    ("jungle", "丛林", "environment", "extended", _both()),
    ("meadow", "草地牧场", "environment", "extended", _both()),
    ("wetland", "湿地", "environment", "extended", _both()),
    ("volcano", "火山", "environment", "extended", _both()),
    ("stream", "溪流", "environment", "extended", _both()),
    ("coral reef", "珊瑚礁", "environment", "extended", _sam3()),
    ("bamboo", "竹子", "environment", "extended", _both()),
    ("moss", "苔藓", "environment", "extended", _sam3()),
    # urban / built
    ("city", "城市", "environment", "extended", _both()),
    ("village", "村庄", "environment", "extended", _both()),
    ("farm", "农场", "environment", "extended", _both()),
    ("garden", "花园", "environment", "extended", _both()),
    ("park", "公园", "environment", "extended", _both()),
    ("playground", "游乐场", "environment", "extended", _both()),
    ("stadium", "体育场", "environment", "extended", _both()),
    ("temple", "寺庙", "environment", "extended", _both()),
    ("church", "教堂", "environment", "extended", _both()),
    ("castle", "城堡", "environment", "extended", _both()),
    ("barn", "谷仓", "environment", "extended", _both()),
    ("skyscraper", "摩天大楼", "environment", "extended", _both()),
    ("skyline", "天际线", "environment", "extended", _both()),
    ("railroad", "铁轨", "environment", "extended", _both()),
    ("tunnel", "隧道", "environment", "extended", _both()),
    ("traffic light", "红绿灯", "environment", "extended", _both()),
    ("street light", "路灯", "environment", "extended", _both()),
    ("billboard", "广告牌", "environment", "extended", _both()),
    ("statue", "雕像", "environment", "extended", _both()),
    ("sign", "标志牌", "environment", "extended", _both()),
    ("awning", "遮阳篷", "environment", "extended", _both()),
    ("railing", "栏杆", "environment", "extended", _both()),
    ("column", "柱子", "environment", "extended", _both()),
    ("pier", "码头", "environment", "extended", _both()),
    ("harbor", "港口", "environment", "extended", _both()),
    ("airport", "机场", "environment", "extended", _both()),
    ("parking lot", "停车场", "environment", "extended", _both()),
    ("construction site", "建筑工地", "environment", "extended", _both()),
    ("crosswalk", "人行横道", "environment", "extended", _both()),
    # indoor rooms / setting (SAM3 segments these better as stuff)
    ("kitchen", "厨房", "environment", "extended", _both()),
    ("bedroom", "卧室", "environment", "extended", _both()),
    ("bathroom", "浴室", "environment", "extended", _both()),
    ("living room", "客厅", "environment", "extended", _both()),
    ("office", "办公室", "environment", "extended", _both()),
    ("restaurant", "餐厅", "environment", "extended", _both()),
    ("shop", "商店", "environment", "extended", _both()),
    ("market", "市场", "environment", "extended", _both()),
    ("classroom", "教室", "environment", "extended", _both()),
    ("fireplace", "壁炉", "environment", "extended", _both()),
    ("counter", "柜台", "environment", "extended", _both()),
    ("stage", "舞台", "environment", "extended", _both()),
    ("tent", "帐篷", "environment", "extended", _both()),
    ("shadow", "影子", "environment", "extended", _sam3()),
    ("reflection", "倒影", "environment", "extended", _sam3()),
]

# ---------------------------------------------------------------------------
# SAM 3 parts — OID human parts + common object parts
# LLMDet boxes these poorly; keep SAM3-only.
# ---------------------------------------------------------------------------
_SAM3_PARTS: list[_ENTRY] = [
    ("face", "脸", "part", "extended", _sam3()),
    ("head", "头", "part", "extended", _sam3()),
    ("hand", "手", "part", "extended", _sam3()),
    ("arm", "手臂", "part", "extended", _sam3()),
    ("leg", "腿", "part", "extended", _sam3()),
    ("foot", "脚", "part", "extended", _sam3()),
    ("hair", "头发", "part", "extended", _sam3()),
    ("eye", "眼睛", "part", "extended", _sam3()),
    ("ear", "耳朵", "part", "extended", _sam3()),
    ("mouth", "嘴", "part", "extended", _sam3()),
    ("wheel", "轮子", "part", "extended", _sam3()),
    ("tire", "轮胎", "part", "extended", _sam3()),
    ("wing", "翅膀", "part", "extended", _sam3()),
    ("tail", "尾巴", "part", "extended", _sam3()),
    ("branch", "树枝", "part", "extended", _sam3()),
    ("tree trunk", "树干", "part", "extended", _sam3()),
]

ENTRIES: list[_ENTRY] = (
    _CORE_SUBJECT
    + _CORE_ENVIRONMENT
    + _EXTENDED_SUBJECT
    + _EXTENDED_ENVIRONMENT
    + _SAM3_PARTS
)


def iter_entries(
    model: Optional[Model] = None,
    role: Optional[Role] = None,
    tier: Literal["core", "extended", "all"] = "all",
) -> Iterable[_ENTRY]:
    for en, zh, r, t, models in ENTRIES:
        if model is not None and model not in models:
            continue
        if role is not None and r != role:
            continue
        if tier != "all" and t != tier:
            continue
        yield en, zh, r, t, models


def get_entry(name: str) -> Optional[dict]:
    """Look up metadata for an English or Chinese name."""
    key = name.strip().lower()
    for en, zh, role, tier, models in ENTRIES:
        if en.lower() == key or zh == name:
            return {
                "en": en,
                "zh": zh,
                "role": role,
                "tier": tier,
                "models": models,
            }
    return None


def get_vocab(
    model: Optional[Model] = None,
    role: Optional[Role] = None,
    tier: Literal["core", "extended", "all"] = "core",
    lang: Literal["en", "zh"] = "en",
) -> list[str]:
    """Return unique prompt names in definition order."""
    seen: set[str] = set()
    out: list[str] = []
    idx = 0 if lang == "en" else 1
    for row in iter_entries(model=model, role=role, tier=tier):
        name = row[idx]
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


def to_llmdet_text(
    names: Optional[Sequence[str]] = None,
    chunk_size: int = 40,
) -> list[str]:
    """Period-separated chunks for LLMDet / Grounding DINO.

    Example: ``"person . car . tree ."``
    Official LVIS demo uses ``--chunked-size 80``; 40 is safer for BERT 256.
    """
    if names is None:
        names = get_vocab(model="llmdet", tier="core")
    chunks: list[str] = []
    buf: list[str] = []
    for name in names:
        buf.append(name)
        if len(buf) >= chunk_size:
            chunks.append(" . ".join(buf) + " .")
            buf = []
    if buf:
        chunks.append(" . ".join(buf) + " .")
    return chunks


def to_sam3_prompts(
    names: Optional[Sequence[str]] = None,
    include_parts: bool = False,
) -> list[str]:
    """Short noun phrases for ``Sam3Processor.set_text_prompt``."""
    if names is not None:
        return list(names)
    names = get_vocab(model="sam3", tier="core")
    if include_parts:
        names = names + get_vocab(model="sam3", role="part", tier="extended")
    return names


# Drop-in for WeDetect-style ``from open_free_vocab import prompt_free_name``.
# English: this is what LLMDet / SAM 3 should be prompted with.
prompt_free_name: list[str] = get_vocab(tier="core", lang="en")
prompt_free_name_zh: list[str] = get_vocab(tier="core", lang="zh")

# Model-specific defaults (core only; parts stay opt-in).
LLMDET_PROMPTS: list[str] = get_vocab(model="llmdet", tier="core")
SAM3_PROMPTS: list[str] = get_vocab(model="sam3", tier="core")


if __name__ == "__main__":
    core = get_vocab(tier="core")
    ext = get_vocab(tier="all")
    print(f"core (default prompt_free_name): {len(core)}")
    print(f"  subject:      {len(get_vocab(role='subject', tier='core'))}")
    print(f"  environment:  {len(get_vocab(role='environment', tier='core'))}")
    print(f"all unique:     {len(ext)}")
    print(f"  llmdet all:   {len(get_vocab(model='llmdet', tier='all'))}")
    print(f"  sam3 all:     {len(get_vocab(model='sam3', tier='all'))}")
    print(f"  sam3 parts:   {len(get_vocab(model='sam3', role='part', tier='extended'))}")
    print("llmdet first chunk:", to_llmdet_text(chunk_size=8)[0])
    print("sam3 sample:", SAM3_PROMPTS[:8])
