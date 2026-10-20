/*
 * SmokeTest.java — short transient solve to verify licenses and coupling.
 * Does not overwrite weldpool_2d.mph (batch -nosave).
 */

import com.comsol.model.*;
import com.comsol.model.util.*;

/** Short solve of the weld-pool model (full physics including Marangoni weak form). */
public class SmokeTest {

  private static final String MPH =
      "C:/Users/Legat/.comsol/v62/applications/files/user/weldpool_2d.mph";

  public static Model run() {
    try {
      Model model = ModelUtil.load("Model", MPH);
      model.param().set("t_end", "0.02[s]");
      model.study("std1").feature("time").set("tlist", "range(0,0.005,0.02)");
      model.sol("sol1").feature("t1").set("tlist", "range(0,0.005,0.02)");
      model.sol("sol1").runAll();
      model.result().numerical("maxT").setResult();
      double[][] tmax = model.result().numerical("maxT").getReal();
      System.out.println("Smoke OK: Tmax=" + tmax[0][0] + " K after 0.02 s");
      return model;
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  public static void main(String[] args) {
    run();
  }
}
