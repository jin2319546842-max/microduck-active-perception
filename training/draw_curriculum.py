"""Regenerate the repository's editable SVG course map from scene templates."""
from pathlib import Path
from html import escape
from .env import TEMPLATES

def main():
    families=['empty','single','enclosure','tee','three_sides','staggered']
    labels=['C0: Goal reaching','C1: Left / right opening','C2: Enclosure divider','C2: T-shaped occlusion','C2: Three-sided obstacles','C3: Repeated decisions']
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="770" viewBox="0 0 1080 770" role="img" aria-label="Curriculum scene layouts: blue start, green goal, orange walls">',
           '<rect width="1080" height="770" fill="#f1f5f7"/>',
           '<text x="24" y="32" font-family="sans-serif" font-size="22" fill="#143947">Microduck: proposed active-perception curriculum</text>']
    for n,family in enumerate(families):
        ox=20+(n%3)*355;oy=60+(n//3)*320;walls,start,goal=TEMPLATES[family]
        parts.append(f'<rect x="{ox}" y="{oy}" width="335" height="300" rx="8" fill="#19364b"/>')
        parts.append(f'<text x="{ox+12}" y="{oy+25}" font-family="sans-serif" font-size="16" fill="white">{escape(labels[n])}</text>')
        def pt(p):return ox+167.5+p[0]*54,oy+162-p[1]*54
        for i in range(-4,5):
            x,_=pt((i*.5,0));_,y=pt((0,i*.5))
            parts.append(f'<path d="M {x} {oy+42} v 240 M {ox+38} {y} h 259" stroke="#355365" stroke-width=".6"/>')
        for x,y,w,h in walls:
            px,py=pt((x-w/2,y+h/2));parts.append(f'<rect x="{px}" y="{py}" width="{w*54}" height="{h*54}" fill="#ff771f"/>')
        for p,color,label in [(start,'#54bdff','S'),(goal,'#22e450','G')]:
            x,y=pt(p);parts.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{color}"/><text x="{x+11}" y="{y+5}" font-family="sans-serif" font-size="14" fill="white">{label}</text>')
    parts.extend(['<text x="24" y="724" font-family="sans-serif" font-size="17" fill="#143947">C4: mix C1-C3 + mirrored layouts + scale variation + ToF noise / dropout</text>',
                  '<text x="24" y="752" font-family="sans-serif" font-size="14" fill="#476575">S = start | G = goal | These are curriculum designs, not training results. No paths are supplied to the policy.</text>','</svg>'])
    path=Path(__file__).parents[1]/'docs/curriculum.svg';path.parent.mkdir(exist_ok=True);path.write_text('\n'.join(parts),encoding='utf8');print(path)

if __name__=='__main__':main()
