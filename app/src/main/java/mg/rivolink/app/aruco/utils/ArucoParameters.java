package mg.rivolink.app.aruco.utils;

import org.opencv.aruco.Aruco;
import org.opencv.aruco.Dictionary;

import java.util.LinkedHashMap;
import java.util.Map;

public class ArucoParameters {

    private static final Map<String, Integer> arucoDictionaries = new LinkedHashMap<>();

    static {
        arucoDictionaries.put("DICT_4X4_50", Aruco.DICT_4X4_50);
        arucoDictionaries.put("DICT_4X4_100", Aruco.DICT_4X4_100);
        arucoDictionaries.put("DICT_4X4_250", Aruco.DICT_4X4_250);
        arucoDictionaries.put("DICT_4X4_1000", Aruco.DICT_4X4_1000);
        arucoDictionaries.put("DICT_5X5_50", Aruco.DICT_5X5_50);
        arucoDictionaries.put("DICT_5X5_100", Aruco.DICT_5X5_100);
        arucoDictionaries.put("DICT_5X5_250", Aruco.DICT_5X5_250);
        arucoDictionaries.put("DICT_5X5_1000", Aruco.DICT_5X5_1000);
        arucoDictionaries.put("DICT_6X6_50", Aruco.DICT_6X6_50);
        arucoDictionaries.put("DICT_6X6_100", Aruco.DICT_6X6_100);
        arucoDictionaries.put("DICT_6X6_250", Aruco.DICT_6X6_250);
        arucoDictionaries.put("DICT_6X6_1000", Aruco.DICT_6X6_1000);
        arucoDictionaries.put("DICT_7X7_50", Aruco.DICT_7X7_50);
        arucoDictionaries.put("DICT_7X7_100", Aruco.DICT_7X7_100);
        arucoDictionaries.put("DICT_7X7_250", Aruco.DICT_7X7_250);
        arucoDictionaries.put("DICT_7X7_1000", Aruco.DICT_7X7_1000);
        arucoDictionaries.put("DICT_ORIGINAL", Aruco.DICT_ARUCO_ORIGINAL);
    }

    // Method to get dictionary ID by name
    public static int getDictionaryIdByName(String name) {
        return arucoDictionaries.getOrDefault(name, Aruco.DICT_4X4_50);
    }

    // Method to get OpenCV dictionary by name
    public static Dictionary getDictionaryByName(String name) {
        int dictionaryId = getDictionaryIdByName(name);
        return Aruco.getPredefinedDictionary(dictionaryId);
    }

    // Method to get all available dictionary names
    public static String[] getAvailableDictionaries() {
        return arucoDictionaries.keySet().toArray(new String[0]);
    }
}
