package edu.rsatu.encryption.utils;

public class StringUtils {

    private static final String[] SUPERSCRIPT_DIGITS = {
            "⁰",
            "¹",
            "²",
            "³",
            "⁴",
            "⁵",
            "⁶",
            "⁷",
            "⁸",
            "⁹"
    };

    public String getSuperscriptDigit(Integer digit) {
        return SUPERSCRIPT_DIGITS[digit];
    }

    public Integer getDigit(String superscriptDigit) {
        for (int i = 0; i < SUPERSCRIPT_DIGITS.length; i++) {
            if (SUPERSCRIPT_DIGITS[i].equals(superscriptDigit)) {
                return i;
            }
        }

        return -1;
    }
}
