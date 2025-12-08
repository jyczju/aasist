import matplotlib.pyplot as plt
from matplotlib import font_manager

# 获取所有支持中文的字体（包含 "Song"、"Hei"、"Kai"、"Fang"、"Microsoft" 等关键词）
chinese_fonts = [f.name for f in font_manager.fontManager.ttflist
                 if any(kw in f.name for kw in ['Song', 'Hei', 'Kai', 'Fang', 'Microsoft', 'PingFang', 'STHeiti'])]

print("可用的中文字体：")
for f in sorted(set(chinese_fonts)):
    print(f)