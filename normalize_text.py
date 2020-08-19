import re
import unicodedata


def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    return s


def remove_extra_spaces(s):
    s = re.sub('[ 　\n\t]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s


def normalize_text(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
                  '！”＃＄％＆’（）＊＋，−．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))
    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]', '', s)  # remove tildes
    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，−．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    s = re.sub(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)", "", s)  # remove URLs
    return s


if __name__ == "__main__":
    assert "0123456789" == normalize_text("０１２３４５６７８９")
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" == normalize_text("ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ")
    assert "abcdefghijklmnopqrstuvwxyz" == normalize_text("ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ")
    assert "!\"#$%&'()*+,-./:;<>?@[¥]^_`{|}" == normalize_text("！”＃＄％＆’（）＊＋，−．／：；＜＞？＠［￥］＾＿｀｛｜｝")
    assert "＝。、・「」" == normalize_text("＝。、・「」")
    assert "ハンカク" == normalize_text("ﾊﾝｶｸ")
    assert "o-o" == normalize_text("o₋o")
    assert "majikaー" == normalize_text("majika━")
    assert "わい" == normalize_text("わ〰い")
    assert "スーパー" == normalize_text("スーパーーーー")
    assert "!#" == normalize_text("!#")
    assert "ゼンカクスペース" == normalize_text("ゼンカク　スペース")
    assert "おお" == normalize_text("お             お")
    assert "おお" == normalize_text("      おお")
    assert "おお" == normalize_text("おお      ")
    assert "検索エンジン自作入門を買いました!!!" == \
        normalize_text("検索 エンジン 自作 入門 を 買い ました!!!")
    assert "アルゴリズムC" == normalize_text("アルゴリズム C")
    assert "PRML副読本" == normalize_text("　　　ＰＲＭＬ　　副　読　本　　　")
    assert "Coding the Matrix" == normalize_text("Coding the Matrix")
    assert "南アルプスの天然水Sparking Lemonレモン一絞り" == \
        normalize_text("南アルプスの　天然水　Ｓｐａｒｋｉｎｇ　Ｌｅｍｏｎ　レモン一絞り")
