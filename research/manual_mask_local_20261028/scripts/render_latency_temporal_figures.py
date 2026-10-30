import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "runs/latency_temporal_v1/analysis"
OUTPUT = ROOT / "figures/latency_temporal_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)
FONT = "/System/Library/Fonts/Helvetica.ttc"


def font(size):
    return ImageFont.truetype(FONT,size) if Path(FONT).exists() else ImageFont.load_default(size=size)


def main():
    latency = json.loads((ANALYSIS/"latency.json").read_text())
    fig = Image.new("RGB",(1500,990),"white");draw=ImageDraw.Draw(fig)
    draw.text((65,35),"F vs FIX32: resident-model end-to-end latency",font=font(32),fill="#14213d")
    draw.text((65,83),"8 seen sources; each bar is the median of 3 repetitions; warmups and profiler excluded",font=font(20),fill="#505b6a")
    left,right,top,bottom=190,1330,160,820
    for tick in range(9):
        x=left+tick*(right-left)/8
        draw.line((x,top-15,x,bottom),fill="#e3e6ea",width=2)
        draw.text((x-5,bottom+15),str(tick),font=font(20),fill="#405060")
    for i,row in enumerate(latency["paired"]):
        y=top+i*80
        draw.text((65,y+17),row["source_group_id"],font=font(22),fill="#14213d")
        for j,(name,color) in enumerate([("F","#8797a8"),("FIX32","#2563b4")]):
            value=row[name]["median_seconds"];yy=y+j*29
            draw.rectangle((left,yy,left+value*(right-left)/8,yy+23),fill=color)
            draw.text((left+value*(right-left)/8+10,yy),f"{value:.3f}",font=font(19),fill=color)
    draw.text((660,bottom+48),"Latency (seconds)",font=font(20),fill="#405060")
    draw.rectangle((65,935,87,957),fill="#8797a8");draw.text((100,933),"F",font=font(20),fill="#14213d")
    draw.rectangle((175,935,197,957),fill="#2563b4");draw.text((210,933),"FIX32",font=font(20),fill="#14213d")
    draw.text((400,933),"Diagnostic speed gain; edit-quality failures retained in all 8 sources",font=font(20),fill="#8b3f28")
    fig.save(OUTPUT/"latency_paired.png")
    components=latency["aggregate"]
    labels=[("dit","DiT", "#2563b4"),("prepare_latents_vae_encode","Source VAE / latent preparation","#47a394"),
            ("vae_decode","VAE decode","#e6a23c"),("prompt_encode","Prompt encoding","#a88ac8")]
    fig=Image.new("RGB",(1500,540),"white");draw=ImageDraw.Draw(fig)
    draw.text((65,35),"Where the time goes",font=font(32),fill="#14213d")
    draw.text((65,85),"CUDA intervals: mean of per-source medians; CPU remainder is not allocated to a module",font=font(20),fill="#505b6a")
    left,right=190,1390
    for i,arm in enumerate(["F","FIX32"]):
        y=155+i*140;draw.text((65,y+25),arm,font=font(26),fill="#14213d")
        x=left
        for key,label,color in labels:
            value=components[arm]["component_mean_source_medians_seconds"][key]
            width=value*(right-left)/7.5
            draw.rectangle((x,y,x+width,y+73),fill=color)
            draw.text((x+8,y+24),f"{value:.3f}s" if width > 90 else f"{value:.2f}s",font=font(18),fill="white")
            x+=width
        total=components[arm]["mean_source_median_seconds"]
        draw.text((left,y+85),f"Measured end-to-end: {total:.3f}s",font=font(20),fill="#14213d")
    for i,(key,label,color) in enumerate(labels):
        x=65+(i%2)*700;y=450+(i//2)*35
        draw.rectangle((x,y,x+18,y+18),fill=color);draw.text((x+30,y-2),label,font=font(20),fill="#14213d")
    fig.save(OUTPUT/"latency_components.png")
    temporal_path=ANALYSIS/"temporal.json"
    if temporal_path.exists():
        temporal=json.loads(temporal_path.read_text());rows=temporal["rows"]
        ids=list(dict.fromkeys(row["source_group_id"] for row in rows))
        fig=Image.new("RGB",(1500,910),"white");draw=ImageDraw.Draw(fig)
        draw.text((65,35),"Equal-budget temporal schedules: full-image MAE to F",font=font(30),fill="#14213d")
        draw.text((65,82),"All 8 sources retained, including the globally redrawn bollard case; lower is better",font=font(20),fill="#505b6a")
        left,right,top,bottom=140,1400,140,755
        max_y=max(row["mae_vs_F"] for row in rows)*1.13
        for tick in range(6):
            y=bottom-tick*(bottom-top)/5
            draw.line((left,y,right,y),fill="#e3e6ea",width=2)
            draw.text((40,y-12),f"{max_y*tick/5:.3f}",font=font(20),fill="#405060")
        colors={"EARLY":"#2563b4","MIDDLE":"#47a394","LATE":"#e6a23c"}
        for i,source in enumerate(ids):
            x=left+(i+.5)*(right-left)/len(ids)
            for j,(arm,color) in enumerate(colors.items()):
                row=next(r for r in rows if r["source_group_id"]==source and r["arm"]==arm)
                y=bottom-row["mae_vs_F"]*(bottom-top)/max_y
                xx=x+(j-1)*32
                draw.rectangle((xx-13,y,xx+13,bottom),fill=color)
            draw.text((x-36,bottom+18),source,font=font(17),fill="#14213d")
        for i,(arm,color) in enumerate(colors.items()):
            x=300+i*300;draw.rectangle((x,835,x+20,855),fill=color);draw.text((x+30,832),arm,font=font(22),fill="#14213d")
        fig.save(OUTPUT/"temporal_all8.png")
    print(OUTPUT)


if __name__=="__main__":main()
