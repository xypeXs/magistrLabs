package edu.rsatu.encryption.utils;

import java.util.ArrayList;
import java.util.List;

public class PolynomialUtils {

    public static String mapBinaryToStringPolynomial(int binaryCoefficients) {
        List<Integer> powerOfTwoList = getNumberPowersOfTwo(binaryCoefficients);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < powerOfTwoList.size(); i++) {
            if (i == powerOfTwoList.size() - 1) {
                sb.append(powerOfTwoList.get(i));
            }
        }

        return sb.toString();
    }

    public static List<Integer> getNumberPowersOfTwo(int number) {
        List<Integer> powerList = new ArrayList<>((int) (Math.log((number))));
        while (number > 0) {
            powerList.add(number % 2);
            number /= 2;
        }

        return powerList;
    }
}
