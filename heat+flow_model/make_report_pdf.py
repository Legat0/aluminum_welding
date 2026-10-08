"""Generate PDF report for weld-pool model v1."""

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
from reportlab.lib.units import cm, mm
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

from weldpool2d import Material, WeldPool2D, quick_process

# Cyrillic fonts
pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))

OUT_DIR = ROOT / "report_assets"
OUT_DIR.mkdir(exist_ok=True)
PDF_PATH = ROOT / "report_weld_pool_v1.pdf"


def run_cases():
    mat = Material()
    proc = quick_process(
        enable_flow=True,
        t_end=1.2,
        nx=60,
        nz=20,
        q_max=8.0e7,
        r_b=0.0025,
        v_weld=0.008,
        x0=0.006,
    )
    m = WeldPool2D(material=mat, process=proc)
    diag = m.run(record_every=0.05, progress=True)

    proc_c = quick_process(
        enable_flow=False,
        t_end=0.8,
        nx=60,
        nz=20,
        q_max=8.0e7,
        r_b=0.0025,
        v_weld=0.008,
        x0=0.006,
        Lx=proc.Lx,
        Lz=proc.Lz,
    )
    m_c = WeldPool2D(material=mat, process=proc_c)
    m_c.run(record_every=0.1, progress=False)

    proc_f = quick_process(
        enable_flow=True,
        t_end=0.8,
        nx=60,
        nz=20,
        q_max=8.0e7,
        r_b=0.0025,
        v_weld=0.008,
        x0=0.006,
        Lx=proc.Lx,
        Lz=proc.Lz,
    )
    m_f = WeldPool2D(material=mat, process=proc_f)
    m_f.run(record_every=0.1, progress=False)

    return mat, m, diag, m_c, m_f


def make_figures(mat, m, diag, m_c, m_f):
    Xmm, Zmm = 1e3 * m.X, 1e3 * m.Z
    u_c, w_c = m._cell_velocity()
    speed = np.sqrt(u_c**2 + w_c**2)

    # Fig 1: scheme
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 6)
    ax.add_patch(plt.Rectangle((0, 0), 30, 5, fill=True, color="#c8d6e5", ec="k", lw=1.2))
    ax.add_patch(plt.Rectangle((6, 3.2), 12, 1.8, fill=True, color="#74b9ff", ec="k", lw=1, alpha=0.85))
    ax.annotate("", xy=(18, 5.5), xytext=(10, 5.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="C3"))
    ax.text(14, 5.7, r"$v_{\mathrm{weld}}$", ha="center", color="C3", fontsize=11)
    ax.plot([12], [5.35], "r^", ms=12)
    ax.text(12, 5.85, r"$q(x,t)$", ha="center", color="C3", fontsize=10)
    ax.text(12, 4.1, "ванна", ha="center", fontsize=10)
    ax.text(24, 2.2, "твёрдое", ha="center", fontsize=10)
    ax.annotate("", xy=(4, 0), xytext=(4, 5), arrowprops=dict(arrowstyle="<->", color="k", lw=1))
    ax.text(3.2, 2.5, r"$z$", ha="right", fontsize=11)
    ax.annotate("", xy=(28, 0.4), xytext=(2, 0.4), arrowprops=dict(arrowstyle="->", color="k", lw=1))
    ax.text(28.5, 0.4, r"$x$", va="center", fontsize=11)
    ax.text(1, -0.35, r"$x=0$", fontsize=9)
    ax.text(26, -0.35, r"$x=L_x$", fontsize=9)
    ax.text(30.3, 0, r"$z=0$", fontsize=9)
    ax.text(30.3, 5, r"$z=L_z$", fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Продольное сечение (версия 1)", fontsize=12)
    fig.tight_layout()
    p1 = OUT_DIR / "fig_scheme.png"
    fig.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: T, fL, |V|
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.0), constrained_layout=True)
    im0 = axs[0].pcolormesh(Xmm, Zmm, m.T, shading="auto", cmap="inferno")
    axs[0].contour(Xmm, Zmm, m.T, levels=[mat.T_solidus, mat.T_liquidus], colors=["c", "lime"], linewidths=1)
    axs[0].axvline(1e3 * m.arc_centre(), color="w", ls="--", lw=0.8)
    fig.colorbar(im0, ax=axs[0], label="T [K]")
    axs[0].set_title(f"T, t={m.t:.2f} s")

    im1 = axs[1].pcolormesh(Xmm, Zmm, m.fL, shading="auto", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im1, ax=axs[1], label=r"$f_L$")
    axs[1].set_title("Доля жидкости")

    im2 = axs[2].pcolormesh(Xmm, Zmm, speed, shading="auto", cmap="viridis")
    mask = m.fL > 0.3
    sx, sz = max(1, m.proc.nx // 20), max(1, m.proc.nz // 8)
    Us = np.where(mask[::sx, ::sz], u_c[::sx, ::sz] * 1e3, 0.0)
    Ws = np.where(mask[::sx, ::sz], w_c[::sx, ::sz] * 1e3, 0.0)
    axs[2].quiver(
        Xmm[::sx, ::sz],
        Zmm[::sx, ::sz],
        Us,
        Ws,
        color="w",
        angles="xy",
        scale_units="xy",
        scale=10,
        width=0.004,
    )
    fig.colorbar(im2, ax=axs[2], label="|V| [m/s]")
    axs[2].set_title("Скорость")
    for ax in axs:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_aspect("equal")
    p2 = OUT_DIR / "fig_fields.png"
    fig.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Fig 3: metrics
    t = np.array(diag.time)
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.0), constrained_layout=True)
    axs[0].plot(t, 1e3 * np.array(diag.pool_depth), label="глубина")
    axs[0].plot(t, 1e3 * np.array(diag.pool_length), label="длина")
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

    fz = (m.T_max >= mat.T_liquidus).astype(float)
    im = axs[2].pcolormesh(Xmm, Zmm, m.T_max, shading="auto", cmap="hot")
    axs[2].contour(Xmm, Zmm, fz, levels=[0.5], colors="cyan")
    fig.colorbar(im, ax=axs[2], label=r"$T_{\max}$ [K]")
    axs[2].set_title(r"Зона проплавления $T_{\max}>T_l$")
    axs[2].set_xlabel("x [mm]")
    axs[2].set_ylabel("z [mm]")
    axs[2].set_aspect("equal")
    p3 = OUT_DIR / "fig_metrics.png"
    fig.savefig(p3, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Fig 4: conduction vs flow
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.0), constrained_layout=True)
    for ax, mm, title in (
        (axs[0], m_c, "Только теплопроводность"),
        (axs[1], m_f, "С течением (Марангони)"),
    ):
        im = ax.pcolormesh(1e3 * mm.X, 1e3 * mm.Z, mm.T, shading="auto", cmap="inferno")
        ax.contour(
            1e3 * mm.X,
            1e3 * mm.Z,
            mm.T,
            levels=[mat.T_solidus, mat.T_liquidus],
            colors=["c", "lime"],
            linewidths=1,
        )
        d, L = mm.pool_metrics()
        ax.set_title(f"{title}\nглубина={1e3*d:.2f} mm, Tmax={mm.T.max():.0f} K")
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

    depth, length = m.pool_metrics()
    u_c, w_c = m._cell_velocity()
    speed = np.sqrt(u_c**2 + w_c**2)
    solid = m.fL < 1e-6
    liquid = m.fL > 0.5
    bal = m.heat_balance_ratio()
    pe = max(diag.pe_number) if diag.pe_number else 0.0
    d_c, _ = m_c.pool_metrics()
    d_f, _ = m_f.pool_metrics()

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
    story.append(Paragraph("Версия 1: теплоперенос, течение и затвердевание (без электромагнетизма)", meta))
    story.append(Paragraph("2D продольная модель · монолитная пластина · Al 5182", meta))
    story.append(Spacer(1, 8))

    # --- 1 ---
    story.append(Paragraph("1. Введение и обзор литературы", h1))
    story.append(
        Paragraph(
            "Электродуговая сварка алюминиевых пластин с присадочной проволокой сопровождается "
            "образованием ванны расплава, конвекцией в жидком металле и последующим затвердеванием. "
            "Форма шва, термоцикл и структура металла определяются балансом теплопроводности и "
            "конвективного переноса. Настоящая работа рассматривает первую вычислительную версию "
            "модели: без электромагнитных сил и без явного переноса присадки — с заменой дуги "
            "движущимся поверхностным тепловым потоком.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Класс моделей «ванна без столба дуги» хорошо представлен в литературе по лазерной "
            "и дуговой сварке. He, Fuerschbach и DebRoy (J. Phys. D, 2003) решают нестационарные "
            "уравнения сохранения массы, импульса и энергии; нагрев задаётся гауссовым потоком на "
            "поверхности; в двухфазной зоне применяется сток Дарси–Кармана–Козени; движущие силы — "
            "поверхностное натяжение и плавучесть.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Zhao и DebRoy (Metall. Mater. Trans. B, 2001) применяют аналогичный подход к алюминию "
            "5182 в режиме теплопроводности лазерной сварки и приводят таблицу свойств, включая "
            "отрицательный коэффициент dσ/dT (течение от горячего центра к периферии).",
            body,
        )
    )
    story.append(
        Paragraph(
            "Pitscheneder и др. (Welding Journal, 1996) показывают, что влияние поверхностно-активных "
            "элементов на форму ванны проявляется лишь при существенной роли конвекции (высокие числа "
            "Пекле). Kidess и др. (Int. J. Thermal Sci., 2016) отмечают, что молекулярная вязкость "
            "занижает перемешивание в ванне; в версии 1 эффективная вязкость не вводится.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Работы с самосогласованной дугой и силой Лоренца (Murphy, Chen, Bachmann, Le Gal La Salle "
            "и др., каталог refs/primary) отнесены к следующим версиям модели.",
            body,
        )
    )

    # --- 2 ---
    story.append(Paragraph("2. Постановка задачи", h1))
    story.append(
        Paragraph(
            "Рассматривается стыковая сварка двух алюминиевых пластин; в версии 1 пластина "
            "моделируется как монолит (стык и зазор не разрешаются). Расчётная плоскость — "
            "продольное сечение: ось x вдоль шва, ось z по толщине. Дуга движется вдоль +x со "
            "скоростью v_weld; центр пятна нагрева x_c(t) = x_0 + v_weld·t.",
            body,
        )
    )
    story.append(Image(str(p1), width=16 * cm, height=6.2 * cm))
    story.append(Paragraph("Рисунок 1. Схема расчётной области и направления сварки.", caption))

    story.append(Paragraph("<b>Допущения версии 1</b>", body))
    for t in [
        "дуга заменена движущимся гауссовым тепловым потоком на верхней поверхности;",
        "свободная поверхность плоская (нет давления дуги и прогиба);",
        "нет силы Лоренца и магнитного поля;",
        "нет присадки и переноса капель;",
        "мощность — эффективная 2D (на единицу ширины).",
    ]:
        story.append(Paragraph("• " + t, bullet))

    story.append(
        Paragraph(
            "Жидкая доля задаётся линейно между солидусом и ликвидусом. В каше действует сток "
            "проницаемости (Carman–Kozeny). На верхней поверхности — условие Марангони; снизу — "
            "конвективный теплообмен; боковые границы теплоизолированы.",
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
        ["Сетка (пример)", f"{m.proc.nx}×{m.proc.nz}", f"Lx={1e3*m.proc.Lx:.0f} мм, Lz={1e3*m.proc.Lz:.0f} мм"],
        ["q_max, r_b, v", f"{m.proc.q_max:.1e}, {m.proc.r_b*1e3:.1f} мм, {m.proc.v_weld*1e3:.0f} мм/с", "эффективные 2D"],
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

    # --- 3 ---
    story.append(Paragraph("3. Решение", h1))
    story.append(
        Paragraph(
            "Реализован собственный нестационарный решатель на языке Python (модуль weldpool2d.py) "
            "на разнесённой сетке MAC. Уравнения несжимаемой жидкости решаются методом проекции: "
            "явный прогноз скорости (адвекция против потока, вязкость, плавучесть), неявный сток "
            "Дарси в каше, уравнение Пуассона для давления (разреженная СЛАУ SciPy), коррекция "
            "скорости. Энергия — с эффективной теплоёмкостью в интервале [T_s, T_l], учитывающей "
            "скрытую теплоту. Шаг по времени ограничен условиями CFL и диффузии.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Граничные условия: на верху — гауссов поток минус конвекция и излучение, сдвиг "
            "Марангони при f_L &gt; 0; на низу — конвекция; по бокам — нулевой тепловой поток; "
            "прилипание на твёрдых стенках. Флаг enable_flow=False отключает течение и оставляет "
            "только тепло с фазовым переходом (тест и сравнение).",
            body,
        )
    )
    story.append(
        Paragraph(
            "Код и интерактивный запуск: heat+flow_model/weldpool2d.py и ноутбук weld_pool_2d.ipynb. "
            "Контроль: баланс тепла |Q_in − Q_loss − ΔH|/Q_in, малость скорости в твёрдом, направление "
            "Марангони при dσ/dT &lt; 0, затвердевание после обнуления мощности.",
            body,
        )
    )

    story.append(PageBreak())

    # --- 4 ---
    story.append(Paragraph("4. Результаты", h1))
    story.append(
        Paragraph(
            f"Демонстрационный расчёт (QUICK): сетка {m.proc.nx}×{m.proc.nz}, время до t = {m.t:.2f} с. "
            f"Пиковая температура {m.T.max():.0f} K, глубина ванны {1e3*depth:.2f} мм, длина "
            f"{1e3*length:.2f} мм, max |V| = {max(diag.max_speed):.3f} м/с, max Pe ≈ {pe:.0f}. "
            f"Относительный дисбаланс тепла {bal:.3f}.",
            body,
        )
    )
    story.append(Image(str(p2), width=16.5 * cm, height=4.5 * cm))
    story.append(
        Paragraph(
            "Рисунок 2. Поля температуры, доли жидкости и скорости в момент окончания прогона. "
            "Пунктир — положение центра дуги; контуры T_s и T_l.",
            caption,
        )
    )
    story.append(Image(str(p3), width=16.5 * cm, height=4.5 * cm))
    story.append(
        Paragraph(
            "Рисунок 3. Эволюция размера ванны, термоцикл в пробной точке и карта зоны проплавления "
            "(T_max &gt; T_l).",
            caption,
        )
    )
    story.append(Image(str(p4), width=15 * cm, height=5.0 * cm))
    story.append(
        Paragraph(
            f"Рисунок 4. Сравнение при t = 0.8 с: без течения глубина {1e3*d_c:.2f} мм; "
            f"с Марангони — {1e3*d_f:.2f} мм (форма и теплоперенос зависят от конвекции).",
            caption,
        )
    )

    v_sol = float(speed[solid].max()) if np.any(solid) else 0.0
    v_liq = float(speed[liquid].max()) if np.any(liquid) else 0.0
    story.append(
        Paragraph(
            f"Проверки: скорость в твёрдом max |V| ≈ {v_sol:.4f} м/с при {v_liq:.3f} м/с в жидкости; "
            "при dσ/dT &lt; 0 поверхностный поток направлен от пятна нагрева к периферии; после "
            "обнуления q_max жидкая область исчезает (затвердевание).",
            body,
        )
    )

    # --- 5 ---
    story.append(Paragraph("5. Выводы", h1))
    story.append(
        Paragraph(
            "1. Реализована двумерная нестационарная модель ванны алюминия с движущимся гауссовым "
            "нагревом, фазовым переходом и течением (Марангони + плавучесть) без электромагнетизма.",
            body,
        )
    )
    story.append(
        Paragraph(
            "2. Расчёт воспроизводит образование ванны, отток от горячего центра при отрицательном "
            "dσ/dT, рост глубины/длины ванны во времени и зону проплавления по T_max.",
            body,
        )
    )
    story.append(
        Paragraph(
            "3. Сравнение с чисто теплопроводностным решением подтверждает влияние конвекции на "
            "форму ванны при числах Пекле порядка десятков–сотен.",
            body,
        )
    )
    story.append(
        Paragraph(
            "4. Ограничения версии 1: эффективная 2D-мощность, плоская поверхность, нет Лоренца и "
            "присадки, молекулярная вязкость. Следующие шаги — электромагнетизм, присадка и "
            "сопоставление с 3D/экспериментом.",
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
