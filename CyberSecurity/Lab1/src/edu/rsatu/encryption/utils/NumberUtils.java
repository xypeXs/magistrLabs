package edu.rsatu.encryption.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class NumberUtils {

    public static String getBinaryString(int number, int minNumberOfBits) {
        StringBuilder sb = new StringBuilder(getBinaryString(number)).reverse();

        sb.append("0".repeat(Math.max(0, minNumberOfBits - sb.length())));

        return sb.reverse().toString();
    }

    public static String getBinaryString(int number) {
        if (number == 0) {
            return "0";
        }

        StringBuilder sb = new StringBuilder();

        while (number > 0) {
            sb.append(number % 2);
            number = number >> 1;
        }

        return sb.reverse().toString();
    }

    public static int getNumberFromBinaryList(List<Integer> binaryNum) {
        int result = 0;
        for (Integer bin : binaryNum) {
            result = (result << 1) + bin;
        }

        return result;
    }

    public static int getBinaryNumber(String binaryNumber) {
        int result = 0;
        for (int i = 0; i < binaryNumber.length(); i++) {
            result = (result << 1) + (binaryNumber.charAt(i) - '0');
        }

        return result;
    }

    public static int bitByBitAndThenXor(int x1, int x2) {
        int result = 0;
        while (x1 > 0 && x2 > 0) {
            result ^= ((x1 & 1) & (x2 & 1));
            x1 = x1 >> 1;
            x2 = x2 >> 1;
        }
        return result;
    }

    public static int modPolynomial(int x1, int divisor) {
        int remainder = x1;
        int remainderBitLength;
        int divisorBitLength = NumberUtils.log2(divisor);
        while ((remainderBitLength = NumberUtils.log2(remainder)) >= divisorBitLength) {
            remainder ^= (divisor << (remainderBitLength - divisorBitLength));
        }

        return remainder;
    }

    public static int log2(int x) {
        return (int) Math.floor(Math.log(x) / Math.log(2));
    }

    public static List<Integer> transposeBinaryMatr(List<Integer> arr) {
        List<Integer> binaryMatr = new ArrayList<>(arr);
        List<Integer> binaryMatrT = new ArrayList<>();
        boolean isReducable = true;
        while (isReducable) {
            isReducable = false;
            int h = 0;
            for (int i = 0; i < binaryMatr.size(); i++) {
                int remainder = binaryMatr.get(i);
                h = (h << 1) + remainder % 2;
                binaryMatr.set(i, remainder >> 1);
                if (binaryMatr.get(i) > 0) {
                    isReducable = true;
                }
            }
            binaryMatrT.add(h);
        }

        Collections.reverse(binaryMatrT);

        return binaryMatrT;
    }

    public static int getXorOfAllBits(int number) {
        int result = 0;

        while (number > 0) {
            result ^= (number & 1);
            number >>= 1;
        }

        return result;
    }
}
