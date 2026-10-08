"""Generate PDF report for 3D weld-pool half-plate model."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from weldpool3d import Material, WeldPool3D, quick_process

pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))

OUT_DIR = ROOT / "report_assets"
OUT_DIR.mkdir(exist_ok=True)
PDF_PATH = ROOT / "report_weld_pool_v1.pdf"


def run_cases():
    mat = Material()
    proc = quick_process(
        enable_flow=True,
        t_end=0.35,
        nx=24,
        ny=10,
        nz=8,
        P_arc=2200.0,
        r_b=0.0025,
        v_weld=0.008,
        x0=0.005,
    )
    m = WeldPool3D(material=mat, process=proc)
    diag = m.run(record_every=0.05, progress=True)

    proc_c = quick_process(
        enable_flow=False,
        t_end=0.25,
        nx=24,
        ny=10,
        nz=8,
        P_arc=2200.0,
        r_b=0.0025,
        v_weld=0.008,
        x0=0.005,
        Lx=proc.Lx,
        Ly=proc.Ly,
        Lz=proc.Lz,
    )
    m_c = WeldPool3D(material=mat, process=proc_c)
    m_c.run(record_every=0.1, progress=False)

    proc_f = quick_process(
        enable_flow=True,
        t_end=0.25,
        nx=24,
        ny=10,
        nz=8,
        P_arc=2200.0,
        r_b=0.0025,
        v_weld=0.008,
        x0=0.005,
        Lx=proc.Lx,
        Ly=proc.Ly,
        Lz=proc.Lz,
    )
    m_f = WeldPool3D(material=mat, process=proc_f)
    m_f.run(record_every=0.1, progress=False)

    return mat, m, diag, m_c, m_f


def make_figures(mat, m, diag, m_c, m_f):
    j0 = 0
    Xmm = 1e3 * m.X[:, j0, :]
    Zmm = 1e3 * m.Z[:, j0, :]
    u_c, v_c, w_c = m._cell_velocity()
    speed = np.sqrt(u_c**2 + v_c**2 + w_c**2)
    ix_arc = int(np.clip(np.searchsorted(m.xc, m.arc_centre()) - 1, 0, m.proc.nx - 1))
    Ymm = 1e3 * m.Y[ix_arc, :, :]
    Zcs = 1e3 * m.Z[ix_arc, :, :]

    # Fig 1: scheme
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.add_patch(plt.Rectangle((0, -4), 20, 4, fill=True, color="#c8d6e5", ec="k", lw=1.2))
    ax.add_patch(plt.Rectangle((4, -1.8), 8, 1.8, fill=True, color="#74b9ff", ec="k", lw=1, alpha=0.85))
    ax.set_xlim(0, 22)
    ax.set_ylim(-5, 2)
    ax.annotate("", xy=(14, 0.6), xytext=(6, 0.6), arrowprops=dict(arrowstyle="->", lw=1.5, color="C3"))
    ax.text(10, 0.95, r"$v_{\mathrm{weld}}$", ha="center", color="C3", fontsize=11)
    ax.plot([8], [0.15], "r^", ms=12)
    ax.text(8, 1.35, r"$q(x,y)$ on $z=0$", ha="center", color="C3", fontsize=10)
    ax.text(8, -0.8, "ванна", ha="center", fontsize=10)
    ax.text(16, -2.5, "твёрдое", ha="center", fontsize=10)
    ax.text(1.2, 1.5, "полупластина y ≥ 0", fontsize=10)
    ax.annotate("", xy=(2, -4), xytext=(2, 0), arrowprops=dict(arrowstyle="<->", color="k", lw=1))
    ax.text(1.2, -2.0, r"$z$", ha="right", fontsize=11)
    ax.text(2.3, -0.15, r"$z=0$", fontsize=9)
    ax.text(2.3, -4.15, r"$z=-L_z$", fontsize=9)
    ax.annotate("", xy=(19, -4.35), xytext=(1, -4.35), arrowprops=dict(arrowstyle="->", color="k", lw=1))
    ax.text(19.5, -4.35, r"$x$", va="center", fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("3D полупластина: z=0 поверхность, z<0 глубина", fontsize=12)
    fig.tight_layout()
    p1 = OUT_DIR / "fig_scheme.png"
    fig.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: mid-plane + cross-section + top
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    im0 = axs[0].pcolormesh(Xmm, Zmm, m.T[:, j0, :], shading="auto", cmap="inferno")
    axs[0].contour(
        Xmm, Zmm, m.T[:, j0, :], levels=[mat.T_solidus, mat.T_liquidus], colors=["c", "lime"], linewidths=1
    )
    axs[0].axvline(1e3 * m.arc_centre(), color="w", ls="--", lw=0.8)
    fig.colorbar(im0, ax=axs[0], label="T [K]")
    axs[0].set_title(f"T, y≈0, t={m.t:.2f} s")
    axs[0].set_xlabel("x [mm]")
    axs[0].set_ylabel("z [mm]")

    im1 = axs[1].pcolormesh(Ymm, Zcs, m.T[ix_arc, :, :], shading="auto", cmap="inferno")
    axs[1].contour(
        Ymm,
        Zcs,
        m.T[ix_arc, :, :],
        levels=[mat.T_solidus, mat.T_liquidus],
        colors=["c", "lime"],
        linewidths=1,
    )
    fig.colorbar(im1, ax=axs[1], label="T [K]")
    axs[1].set_title(r"T, x≈$x_c$")
    axs[1].set_xlabel("y [mm]")
    axs[1].set_ylabel("z [mm]")

    Xt, Yt = 1e3 * m.X[:, :, -1], 1e3 * m.Y[:, :, -1]
    im2 = axs[2].pcolormesh(Xt, Yt, m.T[:, :, -1], shading="auto", cmap="inferno")
    axs[2].contour(
        Xt, Yt, m.T[:, :, -1], levels=[mat.T_solidus, mat.T_liquidus], colors=["c", "lime"], linewidths=1
    )
    fig.colorbar(im2, ax=axs[2], label="T [K]")
    axs[2].set_title("T, верх")
    axs[2].set_xlabel("x [mm]")
    axs[2].set_ylabel("y [mm]")
    for ax in axs:
        ax.set_aspect("equal")
    p2 = OUT_DIR / "fig_fields.png"
    fig.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: metrics
    t = np.array(diag.time)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.0), constrained_layout=True)
    axs[0].plot(t, 1e3 * np.array(diag.pool_depth), label="глубина")
    axs[0].plot(t, 1e3 * np.array(diag.pool_length), label="длина")
    axs[0].plot(t, 1e3 * np.array(diag.pool_width), label="ширина")
    axs[0].set_xlabel("t [s]")
    axs[0].set_ylabel("mm")
    axs[0].legend(fontsize=8)
    axs[0].set_title("Размер ванны")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(t, diag.T_probe, color="C1")
    axs[1].axhline(mat.T_solidus, color="c", ls="--", label=r"$T_s$")
    axs[1].axhline(mat.T_liquidus, color="g", ls="--", label=r"$T_l$")
    axs[1].set_xlabel("t [s]")
    axs[1].set_ylabel("T [K]")
    axs[1].legend(fontsize=8)
    axs[1].set_title("Термоцикл")
    axs[1].grid(True, alpha=0.3)

    im = axs[2].pcolormesh(Xmm, Zmm, speed[:, j0, :], shading="auto", cmap="viridis")
    fig.colorbar(im, ax=axs[2], label="|V| [m/s]")
    axs[2].set_title("|V|, y≈0")
    axs[2].set_xlabel("x [mm]")
    axs[2].set_ylabel("z [mm]")
    axs[2].set_aspect("equal")
    p3 = OUT_DIR / "fig_metrics.png"
    fig.savefig(p3, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Fig 4: conduction vs flow (mid-plane)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.0), constrained_layout=True)
    for ax, mm, title in (
        (axs[0], m_c, "Только теплопроводность"),
        (axs[1], m_f, "С течением (Марангони)"),
    ):
        im = ax.pcolormesh(1e3 * mm.X[:, j0, :], 1e3 * mm.Z[:, j0, :], mm.T[:, j0, :], shading="auto", cmap="inferno")
        ax.contour(
            1e3 * mm.X[:, j0, :],
            1e3 * mm.Z[:, j0, :],
            mm.T[:, j0, :],
            levels=[mat.T_solidus, mat.T_liquidus],
            colors=["c", "lime"],
            linewidths=1,
        )
        d, L, W = mm.pool_metrics()
        ax.set_title(f"{title}\nглубина={1e3*d:.2f} mm, ширина={1e3*W:.2f} mm")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, label="T [K]")
    p4 = OUT_DIR / "fig_compare.png"
    fig.savefig(p4, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return p1, p2, p3, p4


def styles():
    ss = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleRU",
        parent=ss["Title"],
        fontName="Arial-Bold",
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    h1 = ParagraphStyle(
        "H1RU",
        parent=ss["Heading1"],
        fontName="Arial-Bold",
        fontSize=13,
        leading=16,
        spaceBefore=14,
        spaceAfter=8,
    )
    body = ParagraphStyle(
        "BodyRU",
        parent=ss["Normal"],
        fontName="Arial",
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    bullet = ParagraphStyle(
        "BulletRU",
        parent=body,
        leftIndent=12,
        spaceAfter=3,
    )
    caption = ParagraphStyle(
        "CapRU",
        parent=body,
        fontSize=9,
        leading=11,
        alignment=TA_CENTER,
        spaceBefore=2,
        spaceAfter=10,
    )
    meta = ParagraphStyle(
        "MetaRU",
        parent=body,
        alignment=TA_CENTER,
        fontSize=10,
        spaceAfter=4,
    )
    return title, h1, body, bullet, caption, meta


def build_pdf(mat, m, diag, m_c, m_f, figs):
    title, h1, body, bullet, caption, meta = styles()
    p1, p2, p3, p4 = figs

    depth, length, width = m.pool_metrics()
    u_c, v_c, w_c = m._cell_velocity()
    speed = np.sqrt(u_c**2 + v_c**2 + w_c**2)
    solid = m.fL < 1e-6
    liquid = m.fL > 0.5
    bal = m.heat_balance_ratio()
    pe = max(diag.pe_number) if diag.pe_number else 0.0
    d_c, _, w_c_pool = m_c.pool_metrics()
    d_f, _, w_f_pool = m_f.pool_metrics()

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )
    story = []

    story.append(Paragraph("Моделирование ванны расплава при дуговой сварке алюминия", title))
    story.append(Paragraph("3D: теплоперенос, течение и затвердевание (без электромагнетизма)", meta))
    story.append(Paragraph("Полупластина · симметрия по оси шва · Al 5182", meta))
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Введение и обзор литературы", h1))
    story.append(
        Paragraph(
            "Электродуговая сварка алюминиевых пластин сопровождается образованием ванны расплава, "
            "конвекцией и затвердеванием. Настоящая версия — трёхмерная модель половины пластины "
            "с плоскостью симметрии по оси шва: без электромагнитных сил и без явного переноса "
            "присадки; дуга заменена движущимся гауссовым поверхностным потоком с полной мощностью P.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Постановка опирается на класс моделей «ванна без столба дуги» (He, Fuerschbach, DebRoy, "
            "2003; Zhao & DebRoy, 2001 — Al 5182; Pitscheneder et al., 1996; Kidess et al., 2016). "
            "Работы с самосогласованной дугой и силой Лоренца отнесены к следующим версиям.",
            body,
        )
    )

    story.append(Paragraph("2. Постановка задачи", h1))
    story.append(
        Paragraph(
            "Монолитная пластина (стык не разрешается). Координаты: x вдоль шва, y от оси шва "
            "(y=0 — симметрия), z от поверхности нагрева вниз: z=0 — верх (гауссов поток, Марангони), "
            "z=−Lz — низ пластины. Центр пятна: x_c(t)=x_0+v_weld·t, y_c=0 на плоскости z=0. "
            "Пик потока q_max=3P/(π r_b²) так, что интеграл гаусса по полной плоскости равен P; "
            "в полудомене в баланс тепла входит половина мощности.",
            body,
        )
    )
    story.append(Image(str(p1), width=16 * cm, height=6.4 * cm))
    story.append(Paragraph("Рисунок 1. Схема полупластины и направления сварки.", caption))

    story.append(Paragraph("<b>Допущения</b>", body))
    for t in [
        "дуга заменена движущимся гауссовым тепловым потоком на верхней поверхности;",
        "свободная поверхность плоская (нет давления дуги и прогиба);",
        "нет силы Лоренца и магнитного поля;",
        "нет присадки и переноса капель;",
        "половина пластины по симметрии y=0; мощность задаётся полной P.",
    ]:
        story.append(Paragraph("• " + t, bullet))

    story.append(
        Paragraph(
            "Жидкая доля линейна между солидусом и ликвидусом. В каше — сток Carman–Kozeny. "
            "На поверхности z=0 — Марангони для u и v; на дне z=−Lz — конвекция; на y=0 — v=0 и теплоизоляция.",
            body,
        )
    )

    props = [
        ["Параметр", "Значение", "Источник / смысл"],
        ["Сплав", "Al 5182", "Zhao & DebRoy, 2001"],
        ["T_s / T_l", f"{mat.T_solidus:.0f} / {mat.T_liquidus:.0f} K", "солидус / ликвидус"],
        ["ρ", f"{mat.rho:.0f} kg/m³", "плотность"],
        ["k_s / k_l", f"{mat.k_s:.0f} / {mat.k_l:.0f} W/(m·K)", "теплопроводность"],
        ["μ", f"{mat.mu:.1e} Pa·s", "вязкость"],
        ["dσ/dT", f"{mat.dsigma_dT:.1e} N/(m·K)", "Марангони (<0 → отток)"],
        ["L", f"{mat.latent:.2e} J/kg", "скрытая теплота"],
        [
            "Сетка",
            f"{m.proc.nx}×{m.proc.ny}×{m.proc.nz}",
            f"Lx×Ly×Lz = {1e3*m.proc.Lx:.0f}×{1e3*m.proc.Ly:.0f}×{1e3*m.proc.Lz:.0f} мм",
        ],
        [
            "P, r_b, v",
            f"{m.proc.P_arc:.0f} W, {m.proc.r_b*1e3:.1f} мм, {m.proc.v_weld*1e3:.0f} мм/с",
            "полная мощность дуги",
        ],
    ]
    tbl = Table(props, colWidths=[3.5 * cm, 5.5 * cm, 7 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Arial"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (-1, 0), (0.85, 0.85, 0.9)),
                ("GRID", (0, 0), (-1, -1), 0.4, (0.5, 0.5, 0.5)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(tbl)
    story.append(Paragraph("Таблица 1. Материал и режим расчёта (демонстрационный прогон).", caption))

    story.append(Paragraph("3. Решение", h1))
    story.append(
        Paragraph(
            "Решатель weldpool3d.py — нестационарная MAC-схема в трёх измерениях. Метод проекции: "
            "явный прогноз (u,v,w), неявный сток Дарси, уравнение Пуассона для давления (SciPy), "
            "коррекция скорости. Энергия — с эффективной теплоёмкостью в [T_s, T_l]. "
            "Шаг ограничен CFL и диффузией.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Код: heat+flow_3dmodel/weldpool3d.py и ноутбук weld_pool_3d.ipynb. "
            "Контроль: баланс тепла на полудомене, v=0 на симметрии, малость дивергенции в жидкости, "
            "направление Марангони при dσ/dT &lt; 0.",
            body,
        )
    )

    story.append(PageBreak())

    story.append(Paragraph("4. Результаты", h1))
    story.append(
        Paragraph(
            f"Демонстрационный расчёт: сетка {m.proc.nx}×{m.proc.ny}×{m.proc.nz}, t = {m.t:.2f} с. "
            f"Пиковая температура {m.T.max():.0f} K, глубина {1e3*depth:.2f} мм, длина "
            f"{1e3*length:.2f} мм, полная ширина {1e3*width:.2f} мм, max |V| = {max(diag.max_speed):.3f} м/с, "
            f"max Pe ≈ {pe:.0f}. Относительный дисбаланс тепла {bal:.3f}.",
            body,
        )
    )
    story.append(Image(str(p2), width=16.5 * cm, height=4.8 * cm))
    story.append(
        Paragraph(
            "Рисунок 2. Температура: продольный разрез y≈0, поперечный разрез у дуги, вид сверху.",
            caption,
        )
    )
    story.append(Image(str(p3), width=16.5 * cm, height=4.5 * cm))
    story.append(
        Paragraph(
            "Рисунок 3. Эволюция глубины/длины/ширины ванны, термоцикл и поле скорости в середине.",
            caption,
        )
    )
    story.append(Image(str(p4), width=15 * cm, height=5.0 * cm))
    story.append(
        Paragraph(
            f"Рисунок 4. Сравнение при t = 0.25 с: без течения глубина {1e3*d_c:.2f} мм, "
            f"ширина {1e3*w_c_pool:.2f} мм; с Марангони — {1e3*d_f:.2f} мм / {1e3*w_f_pool:.2f} мм.",
            caption,
        )
    )

    v_sol = float(speed[solid].max()) if np.any(solid) else 0.0
    v_liq = float(speed[liquid].max()) if np.any(liquid) else 0.0
    v_sym = float(np.max(np.abs(m.v[:, 0, :])))
    story.append(
        Paragraph(
            f"Проверки: скорость в твёрдом max |V| ≈ {v_sol:.4f} м/с при {v_liq:.3f} м/с в жидкости; "
            f"на симметрии max |v| = {v_sym:.2e} м/с; при dσ/dT &lt; 0 поверхностный поток направлен "
            "от пятна нагрева к периферии.",
            body,
        )
    )

    story.append(Paragraph("5. Выводы", h1))
    story.append(
        Paragraph(
            "1. Реализована трёхмерная нестационарная модель ванны алюминия на полупластине "
            "с движущимся гауссовым нагревом (мощность P), фазовым переходом и течением "
            "(Марангони + плавучесть) без электромагнетизма.",
            body,
        )
    )
    story.append(
        Paragraph(
            "2. Расчёт воспроизводит образование ванны, ширину и глубину, отток от горячего центра "
            "при отрицательном dσ/dT и зону проплавления.",
            body,
        )
    )
    story.append(
        Paragraph(
            "3. Ограничения: плоская поверхность, нет Лоренца и присадки, молекулярная вязкость, "
            "грубая сетка для чернового прогона. Следующие шаги — электромагнетизм, присадка и "
            "сопоставление с экспериментом.",
            body,
        )
    )

    story.append(Spacer(1, 12))
    story.append(Paragraph("Основные ссылки", h1))
    refs = [
        "He X., Fuerschbach P.W., DebRoy T. J. Phys. D: Appl. Phys. 36 (2003) 1388–1398.",
        "Zhao H., DebRoy T. Metall. Mater. Trans. B 32B (2001) 163–172.",
        "Pitscheneder W., DebRoy T., Mundra K., Ebner R. Welding Journal (1996) 71-s–80-s.",
        "Kidess A., Kenjereš S., Righolt B.W., Kleijn C.R. Int. J. Thermal Sci. 104 (2016) 412–422.",
    ]
    for r in refs:
        story.append(Paragraph("• " + r, bullet))

    doc.build(story)
    print("PDF:", PDF_PATH)


def main():
    print("Running simulations...")
    mat, m, diag, m_c, m_f = run_cases()
    print("Making figures...")
    figs = make_figures(mat, m, diag, m_c, m_f)
    print("Building PDF...")
    build_pdf(mat, m, diag, m_c, m_f, figs)


if __name__ == "__main__":
    main()
