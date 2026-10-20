/*
 * WeldPool2D.java — COMSOL 6.2 model matching heat+flow_2d (Al 5182).
 *
 * 2D longitudinal plate: moving Gaussian surface flux, phase change,
 * Marangoni + Boussinesq, Darcy mush. No EM / arc pressure / filler.
 *
 * Coordinates: x along weld, y through thickness (Python z).
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Model builder for the 2D aluminum weld-pool case. */
public class WeldPool2D {

  /** Directory where the MPH is written (must be under COMSOL user files). */
  public static final String OUT_DIR =
      "C:/Users/Legat/.comsol/v62/applications/files/user/";

  public static final String MPH_NAME = "weldpool_2d.mph";

  public static Model run() {
    Model model = build();
    try {
      model.save(OUT_DIR + MPH_NAME);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    return model;
  }

  public static Model build() {
    Model model = ModelUtil.create("Model");

    model.modelPath(OUT_DIR);
    model.label("weldpool_2d");
    model.title("2D weld pool: heat, flow, solidification (Al 5182)");
    model.description(
        "COMSOL 6.2 counterpart of heat+flow_2d/weldpool2d.py. "
            + "Moving Gaussian surface heat flux, linear mushy zone with "
            + "Darcy–Carman–Kozeny sink, Marangoni and Boussinesq buoyancy. "
            + "No electromagnetism, arc pressure, free-surface deformation, or filler.");

    model.component().create("comp1", true);
    model.component("comp1").geom().create("geom1", 2);
    model.component("comp1").mesh().create("mesh1");

    // --- Parameters (Process / Material defaults, not QUICK) ---
    model.param().set("Lx", "40[mm]", "Plate length along weld");
    model.param().set("Lz", "6[mm]", "Plate thickness (y)");
    // Do not use names nx/ny/nz — they clash with COMSOL boundary normals.
    model.param().set("nx_mesh", "120", "Mapped elements along x");
    model.param().set("nz_mesh", "30", "Mapped elements through thickness");
    model.param().set("q_max", "6e7[W/m^2]", "Peak Gaussian heat flux (2D effective)");
    model.param().set("rb", "3[mm]", "Gaussian radius (e^{-3} at r=rb)");
    model.param().set("v_weld", "5[mm/s]", "Welding speed (+x)");
    model.param().set("x0", "8[mm]", "Arc centre at t=0");
    model.param().set("T0", "300[K]", "Initial temperature");
    model.param().set("Ta", "300[K]", "Ambient temperature");
    model.param().set("h_top", "20[W/(m^2*K)]", "Top convection coefficient");
    model.param().set("h_bot", "50[W/(m^2*K)]", "Bottom convection coefficient");
    model.param().set("eps_rad", "0.4", "Surface emissivity");
    model.param().set("t_end", "4[s]", "End time");
    model.param().set("Ts", "850[K]", "Solidus (Al 5182)");
    model.param().set("Tl", "911[K]", "Liquidus (Al 5182)");
    model.param().set("rho", "2300[kg/m^3]", "Density");
    model.param().set("cp_s", "900[J/(kg*K)]", "Solid heat capacity");
    model.param().set("cp_l", "1200[J/(kg*K)]", "Liquid heat capacity");
    model.param().set("k_s", "168[W/(m*K)]", "Solid conductivity");
    model.param().set("k_l", "108[W/(m*K)]", "Liquid conductivity");
    model.param().set("mu", "1.1e-3[Pa*s]", "Dynamic viscosity");
    model.param().set("dsigma_dT", "-3.5e-4[N/(m*K)]", "d(sigma)/dT (Marangoni)");
    model.param().set("L_latent", "3.95e5[J/kg]", "Latent heat");
    model.param().set("beta", "1e-4[1/K]", "Thermal expansion");
    model.param().set("Tref", "911[K]", "Boussinesq reference (liquidus)");
    model.param().set("mushC", "1e6[kg/(m^3*s)]", "Darcy mush coefficient C");
    model.param().set("mushB", "1e-2", "Darcy mush regularisation B");
    model.param().set("A_max", "1e8[kg/(m^3*s)]", "Cap on Darcy coefficient (stability)");
    model.param().set("t_ramp", "0.05[s]", "Heat-flux ramp time at arc start");
    model.param().set("g_acc", "9.81[m/s^2]", "Gravity magnitude (not g_const — reserved)");

    // --- Geometry ---
    model.component("comp1").geom("geom1").create("r1", "Rectangle");
    model.component("comp1").geom("geom1").feature("r1")
        .set("size", new String[]{"Lx", "Lz"});
    model.component("comp1").geom("geom1").run();

    // Boundaries for Rectangle: 1=bottom, 2=right, 3=top, 4=left
    model.component("comp1").selection().create("bndBottom", "Explicit");
    model.component("comp1").selection("bndBottom").geom("geom1", 1);
    model.component("comp1").selection("bndBottom").set(1);
    model.component("comp1").selection("bndBottom").label("Bottom");

    model.component("comp1").selection().create("bndRight", "Explicit");
    model.component("comp1").selection("bndRight").geom("geom1", 1);
    model.component("comp1").selection("bndRight").set(2);
    model.component("comp1").selection("bndRight").label("Right");

    model.component("comp1").selection().create("bndTop", "Explicit");
    model.component("comp1").selection("bndTop").geom("geom1", 1);
    model.component("comp1").selection("bndTop").set(3);
    model.component("comp1").selection("bndTop").label("Top");

    model.component("comp1").selection().create("bndLeft", "Explicit");
    model.component("comp1").selection("bndLeft").geom("geom1", 1);
    model.component("comp1").selection("bndLeft").set(4);
    model.component("comp1").selection("bndLeft").label("Left");

    model.component("comp1").selection().create("bndSides", "Explicit");
    model.component("comp1").selection("bndSides").geom("geom1", 1);
    model.component("comp1").selection("bndSides").set(2, 4);
    model.component("comp1").selection("bndSides").label("Sides");

    // --- Structured mapped mesh 120 x 30 ---
    // Create two opposite-edge distributions with identical counts, then Map.
    // Rectangle edges: 1=bottom, 2=right, 3=top, 4=left (COMSOL default).
    model.component("comp1").mesh("mesh1").create("map1", "Map");
    model.component("comp1").mesh("mesh1").feature("map1").selection().geom("geom1", 2);
    model.component("comp1").mesh("mesh1").feature("map1").selection().set(1);
    model.component("comp1").mesh("mesh1").feature("map1").create("size1", "Size");
    model.component("comp1").mesh("mesh1").feature("map1").feature("size1")
        .set("custom", true);
    model.component("comp1").mesh("mesh1").feature("map1").feature("size1")
        .set("hmaxactive", true);
    model.component("comp1").mesh("mesh1").feature("map1").feature("size1")
        .set("hmax", "Lx/nx_mesh");
    model.component("comp1").mesh("mesh1").feature("map1").feature("size1")
        .set("hminactive", true);
    model.component("comp1").mesh("mesh1").feature("map1").feature("size1")
        .set("hmin", "Lz/nz_mesh");
    model.component("comp1").mesh("mesh1").run();

    // --- Variables ---
    model.component("comp1").variable().create("var1");
    model.component("comp1").variable("var1")
        .set("xc", "x0+v_weld*t");
    model.component("comp1").variable("var1")
        .descr("xc", "Moving arc centre");
    model.component("comp1").variable("var1")
        .set("q_arc", "q_max*min(1,t/t_ramp)*exp(-3*((x-xc)/rb)^2)");
    model.component("comp1").variable("var1")
        .descr("q_arc", "Gaussian surface heat flux (ramped)");
    model.component("comp1").variable("var1")
        .set("fL", "min(1,max(0,(T-Ts)/(Tl-Ts)))");
    model.component("comp1").variable("var1")
        .descr("fL", "Linear liquid fraction");
    model.component("comp1").variable("var1")
        .set("A_mush", "min(mushC*(1-fL)^2/(fL^3+mushB),A_max)");
    model.component("comp1").variable("var1")
        .descr("A_mush", "Darcy–Carman–Kozeny coefficient (capped)");
    // Tangential Marangoni stress as in weldpool2d.py: tau = fL*(dσ/dT)*∂T/∂x
    model.component("comp1").variable("var1")
        .set("tau_mar", "fL*dsigma_dT*Tx");
    model.component("comp1").variable("var1")
        .descr("tau_mar", "Marangoni shear on top (Python BC)");

    // --- Materials: solid and liquid phases for PCM ---
    model.component("comp1").material().create("mat_s", "Common");
    model.component("comp1").material("mat_s").label("Al 5182 solid");
    model.component("comp1").material("mat_s").propertyGroup("def")
        .set("density", new String[]{"rho"});
    model.component("comp1").material("mat_s").propertyGroup("def")
        .set("dynamicviscosity", new String[]{"mu"});
    model.component("comp1").material("mat_s").propertyGroup("def")
        .set("thermalconductivity", new String[]{"k_s"});
    model.component("comp1").material("mat_s").propertyGroup("def")
        .set("heatcapacity", new String[]{"cp_s"});

    model.component("comp1").material().create("mat_l", "Common");
    model.component("comp1").material("mat_l").label("Al 5182 liquid");
    model.component("comp1").material("mat_l").propertyGroup("def")
        .set("density", new String[]{"rho"});
    model.component("comp1").material("mat_l").propertyGroup("def")
        .set("dynamicviscosity", new String[]{"mu"});
    model.component("comp1").material("mat_l").propertyGroup("def")
        .set("thermalconductivity", new String[]{"k_l"});
    model.component("comp1").material("mat_l").propertyGroup("def")
        .set("heatcapacity", new String[]{"cp_l"});

    // Domain material (PCM phases supply k/cp; flow uses user-defined mu/rho below)
    model.component("comp1").material("mat_s").selection().all();

    // --- Laminar Flow ---
    model.component("comp1").physics().create("spf", "LaminarFlow", "geom1");
    model.component("comp1").physics("spf").prop("PhysicalModelProperty")
        .set("Compressibility", "Incompressible");
    // Gravity via NITF Boussinesq only in liquid (custom force); avoids stiff
    // solid buoyancy Jacobian at t=0.
    model.component("comp1").physics("spf").prop("PhysicalModelProperty")
        .set("IncludeGravity", false);
    model.component("comp1").physics("spf").feature("fp1")
        .set("rho_mat", "userdef");
    model.component("comp1").physics("spf").feature("fp1").set("rho", "rho");
    model.component("comp1").physics("spf").feature("fp1")
        .set("mu_mat", "userdef");
    model.component("comp1").physics("spf").feature("fp1").set("mu", "mu");

    // Default wall on all boundaries; override top to Slip + Marangoni shear
    model.component("comp1").physics("spf").create("wallTop", "WallBC", 1);
    model.component("comp1").physics("spf").feature("wallTop")
        .selection().named("bndTop");
    model.component("comp1").physics("spf").feature("wallTop")
        .set("BoundaryCondition", "Slip");

    // Weak form of Python BC: μ∂u/∂n = fL*(dσ/dT)*∂T/∂x  (no Marangoni MP —
    // that applies shear even in solid and causes start-up singularity).
    model.component("comp1").physics("spf").create("wconMar", "WeakContribution", 1);
    model.component("comp1").physics("spf").feature("wconMar")
        .selection().named("bndTop");
    model.component("comp1").physics("spf").feature("wconMar")
        .set("weakExpression", "test(u)*tau_mar");
    model.component("comp1").physics("spf").feature("wconMar")
        .label("Marangoni shear (fL)");

    model.component("comp1").physics("spf").create("prpc1", "PressurePointConstraint", 0);
    model.component("comp1").physics("spf").feature("prpc1").selection().set(1);

    // Darcy + liquid-only Boussinesq (spatial vector length 3 even in 2D)
    model.component("comp1").physics("spf").create("vf1", "VolumeForce", 2);
    model.component("comp1").physics("spf").feature("vf1").selection().all();
    model.component("comp1").physics("spf").feature("vf1")
        .set("F", new String[]{
            "-A_mush*u",
            "-A_mush*v+fL*rho*g_acc*beta*(T-Tref)",
            "0"});
    model.component("comp1").physics("spf").feature("vf1").label("Darcy + buoyancy");

    // --- Heat Transfer in Fluids ---
    model.component("comp1").physics().create("ht", "HeatTransferInFluids", "geom1");
    model.component("comp1").physics("ht").prop("PhysicalModelProperty")
        .set("dz", "1[m]");
    model.component("comp1").physics("ht").prop("PhysicalModelProperty")
        .set("Tref", "Tref");
    model.component("comp1").physics("ht").feature("init1").set("Tinit", "T0");

    // Phase change (apparent heat capacity)
    model.component("comp1").physics("ht").feature("fluid1")
        .create("phc1", "PhaseChangeMaterial", 2);
    model.component("comp1").physics("ht").feature("fluid1").feature("phc1")
        .set("T_pc12", "(Ts+Tl)/2");
    model.component("comp1").physics("ht").feature("fluid1").feature("phc1")
        .set("dT_pc12", "Tl-Ts");
    model.component("comp1").physics("ht").feature("fluid1").feature("phc1")
        .set("L_pc12", "L_latent");
    model.component("comp1").physics("ht").feature("fluid1").feature("phc1")
        .set("MaterialPhase1", "mat_s");
    model.component("comp1").physics("ht").feature("fluid1").feature("phc1")
        .set("MaterialPhase2", "mat_l");
    model.component("comp1").physics("ht").feature("fluid1").feature("phc1")
        .label("Mushy zone Al 5182");

    // Top: Gaussian inward flux + convection + radiation
    model.component("comp1").physics("ht").create("hfTop", "HeatFluxBoundary", 1);
    model.component("comp1").physics("ht").feature("hfTop")
        .selection().named("bndTop");
    model.component("comp1").physics("ht").feature("hfTop")
        .set("HeatFluxType", "GeneralInwardHeatFlux");
    model.component("comp1").physics("ht").feature("hfTop")
        .set("q0_input", "q_arc");
    model.component("comp1").physics("ht").feature("hfTop").label("Arc Gaussian flux");

    model.component("comp1").physics("ht").create("hfTopConv", "HeatFluxBoundary", 1);
    model.component("comp1").physics("ht").feature("hfTopConv")
        .selection().named("bndTop");
    model.component("comp1").physics("ht").feature("hfTopConv")
        .set("HeatFluxType", "ConvectiveHeatFlux");
    model.component("comp1").physics("ht").feature("hfTopConv").set("h", "h_top");
    model.component("comp1").physics("ht").feature("hfTopConv")
        .set("Text_src", "userdef");
    model.component("comp1").physics("ht").feature("hfTopConv").set("Text", "Ta");
    model.component("comp1").physics("ht").feature("hfTopConv").label("Top convection");

    model.component("comp1").physics("ht").create("sarTop", "SurfaceToAmbientRadiation", 1);
    model.component("comp1").physics("ht").feature("sarTop")
        .selection().named("bndTop");
    model.component("comp1").physics("ht").feature("sarTop")
        .set("epsilon_rad_mat", "userdef");
    model.component("comp1").physics("ht").feature("sarTop")
        .set("epsilon_rad", "eps_rad");
    model.component("comp1").physics("ht").feature("sarTop")
        .set("Tamb_src", "userdef");
    model.component("comp1").physics("ht").feature("sarTop").set("Tamb", "Ta");
    model.component("comp1").physics("ht").feature("sarTop").label("Top radiation");

    // Bottom convection
    model.component("comp1").physics("ht").create("hfBot", "HeatFluxBoundary", 1);
    model.component("comp1").physics("ht").feature("hfBot")
        .selection().named("bndBottom");
    model.component("comp1").physics("ht").feature("hfBot")
        .set("HeatFluxType", "ConvectiveHeatFlux");
    model.component("comp1").physics("ht").feature("hfBot").set("h", "h_bot");
    model.component("comp1").physics("ht").feature("hfBot")
        .set("Text_src", "userdef");
    model.component("comp1").physics("ht").feature("hfBot").set("Text", "Ta");
    model.component("comp1").physics("ht").feature("hfBot").label("Bottom convection");

    // Sides: default thermal insulation

    // --- Multiphysics ---
    model.component("comp1").multiphysics().create("nitf1", "NonIsothermalFlow", 2);
    model.component("comp1").multiphysics("nitf1").set("Fluid_physics", "spf");
    model.component("comp1").multiphysics("nitf1").set("Heat_physics", "ht");
    // Couple T↔u; constant density in NITF (alphap=0). Buoyancy only via vf1*fL.
    model.component("comp1").multiphysics("nitf1")
        .set("BoussinesqApproximation", false);
    model.component("comp1").multiphysics("nitf1")
        .set("SpecifyDensity", "CustomLinearizedDensity");
    model.component("comp1").multiphysics("nitf1").set("rhoref", "rho");
    model.component("comp1").multiphysics("nitf1").set("alphap", "0");
    model.component("comp1").multiphysics("nitf1")
        .set("includeViscousDissipation", false);

    // --- Study: time-dependent ---
    model.study().create("std1");
    model.study("std1").label("Transient weld pass");
    model.study("std1").create("time", "Transient");
    model.study("std1").feature("time")
        .setSolveFor("/physics/spf", true);
    model.study("std1").feature("time")
        .setSolveFor("/physics/ht", true);
    model.study("std1").feature("time")
        .setSolveFor("/multiphysics/nitf1", true);
    model.study("std1").feature("time")
        .set("tlist", "range(0,0.05,t_end)");
    model.study("std1").feature("time").set("usertol", true);
    model.study("std1").feature("time").set("rtol", "5e-3");

    model.sol().create("sol1");
    model.sol("sol1").study("std1");
    model.sol("sol1").create("st1", "StudyStep");
    model.sol("sol1").feature("st1").set("study", "std1");
    model.sol("sol1").feature("st1").set("studystep", "time");
    model.sol("sol1").create("v1", "Variables");
    model.sol("sol1").feature("v1").set("control", "time");
    model.sol("sol1").create("t1", "Time");
    model.sol("sol1").feature("t1").set("tlist", "range(0,0.05,t_end)");
    model.sol("sol1").feature("t1").set("initialstepbdfactive", true);
    model.sol("sol1").feature("t1").set("initialstepbdf", "1e-4");
    model.sol("sol1").feature("t1").set("maxstepbdfactive", true);
    model.sol("sol1").feature("t1").set("maxstepbdf", "0.05");
    model.sol("sol1").feature("t1").set("control", "time");
    model.sol("sol1").feature("t1").create("fc1", "FullyCoupled");
    model.sol("sol1").feature("t1").create("d1", "Direct");
    model.sol("sol1").feature("t1").feature("d1").set("linsolver", "pardiso");
    model.sol("sol1").feature("t1").feature("fc1").set("linsolver", "d1");
    model.sol("sol1").feature("t1").feature("fc1").set("dtech", "const");
    model.sol("sol1").feature("t1").feature("fc1").set("damp", 0.7);
    model.sol("sol1").feature("t1").feature("fc1").set("maxiter", 75);
    model.sol("sol1").feature("t1").feature().remove("fcDef");
    model.sol("sol1").attach("std1");

    // --- Probes / derived values ---
    // dset1 is created automatically when sol1 is attached to std1
    // In 2D, domain max uses MaxSurface
    model.result().numerical().create("maxT", "MaxSurface");
    model.result().numerical("maxT").label("Tmax");
    model.result().numerical("maxT").selection().all();
    model.result().numerical("maxT").set("expr", new String[]{"T"});
    model.result().numerical("maxT").set("descr", new String[]{"Maximum temperature"});

    model.result().numerical().create("maxU", "MaxSurface");
    model.result().numerical("maxU").label("|u|max");
    model.result().numerical("maxU").selection().all();
    model.result().numerical("maxU").set("expr", new String[]{"spf.U"});
    model.result().numerical("maxU").set("descr", new String[]{"Maximum speed"});

    model.result().table().create("tbl1", "Table");
    model.result().table("tbl1").label("Probe table");

    // --- Plots ---
    model.result().create("pgT", "PlotGroup2D");
    model.result("pgT").label("Temperature");
    model.result("pgT").create("surf1", "Surface");
    model.result("pgT").feature("surf1").set("expr", "T");
    model.result("pgT").feature("surf1").set("colortable", "ThermalLight");
    model.result("pgT").create("con1", "Contour");
    model.result("pgT").feature("con1").set("expr", "T");
    model.result("pgT").feature("con1").set("levelmethod", "levels");
    model.result("pgT").feature("con1").set("levels", "Tl");
    model.result("pgT").feature("con1").set("coloring", "uniform");
    model.result("pgT").feature("con1").set("color", "black");
    model.result("pgT").feature("con1").label("Liquidus");

    model.result().create("pgFL", "PlotGroup2D");
    model.result("pgFL").label("Liquid fraction fL");
    model.result("pgFL").create("surf1", "Surface");
    model.result("pgFL").feature("surf1").set("expr", "fL");
    model.result("pgFL").feature("surf1").set("colortable", "Rainbow");

    model.result().create("pgVel", "PlotGroup2D");
    model.result("pgVel").label("Velocity and liquidus");
    model.result("pgVel").create("surf1", "Surface");
    model.result("pgVel").feature("surf1").set("expr", "T");
    model.result("pgVel").feature("surf1").set("colortable", "ThermalLight");
    model.result("pgVel").create("arws1", "ArrowSurface");
    model.result("pgVel").feature("arws1").set("expr", new String[]{"u", "v"});
    model.result("pgVel").create("con1", "Contour");
    model.result("pgVel").feature("con1").set("expr", "T");
    model.result("pgVel").feature("con1").set("levelmethod", "levels");
    model.result("pgVel").feature("con1").set("levels", "Tl");
    model.result("pgVel").feature("con1").set("coloring", "uniform");
    model.result("pgVel").feature("con1").set("color", "black");

    return model;
  }

  public static void main(String[] args) {
    run();
  }
}
