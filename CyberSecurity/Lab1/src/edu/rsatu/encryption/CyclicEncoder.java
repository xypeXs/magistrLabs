package edu.rsatu.encryption;

import edu.rsatu.encryption.utils.Pair;

import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class CyclicEncoder {

    private String informationalSequence;
    private int m;

    private static final Map<Integer, Set<Pair<Integer, String>>> IRREDUCIBLE_POLYNOMIAL_MAP = Map.of(
            1, Set.of(Pair.of(3, "x + 1")), // 1 - 11
            2, Set.of(Pair.of(7, "x² + x + 1")), // 2 - 111
            3, Set.of(Pair.of(11, "x³ + x + 1"), Pair.of(13, "x³ + x² + 1")), // 3 - 1011, 1101
            4, Set.of(Pair.of(19, "x⁴ + x + 1"), Pair.of(25, "x⁴ + x³ + 1"), Pair.of(31, "x⁴ + x³ + x² + x + 1")), // 4 - 10011, 11001, 11111
            5, Set.of(Pair.of(37, "x⁵ + x2 + 1"), Pair.of(41, "x⁵ + x³ + 1"), Pair.of(47, "x⁵ + x³ + x² + x + 1"), Pair.of(55, "x⁵ + x⁴ + x² + x + 1"), Pair.of(59, "x⁵ + x⁴ + x³ + x + 1"), Pair.of(61, "x⁵ + x⁴ + x³ + x² + 1")) // 5 - 100101, 101001, 101111, 110111, 111011, 111101


    );

    public static CyclicEncoder from(String informationalSequence) {
        CyclicEncoder encoder = new CyclicEncoder();
        encoder.informationalSequence = informationalSequence;
        encoder.init();
        return encoder;
    }

    private CyclicEncoder() {
    }

    private void init() {
        initM();

    }

    private void initM() {
        if (getK() == 0) {
            m = 1;
        } else if (getK() == 1) {
            m = 2;
        } else if (getK() > 1 && getK() <= 4) {
            m = 3;
        } else if (getK() > 4 && getK() <= 11) {
            m = 4;
        } else {
            m = 5;
        }
    }

    public int getK() {
        return informationalSequence.length();
    }

    public int getM() {
        return this.m;
    }

    public int getN() {
        return getM() + getK();
    }

    public Set<Pair<Integer, String>> getGxSet() {
        return IRREDUCIBLE_POLYNOMIAL_MAP.get(getM());
    }

    public List<Integer> getRemainderList(String gx) {
        int gxValue = getGxValue(gx);
        int seqVal = getSequenceValue();
        int errVal = 1;
        Set<Integer> remainderSet = new HashSet<>(getN());
        for (int i = 0; i < getN(); i++) {
            int errSeqVal = seqVal ^ errVal;
            remainderSet.add(errSeqVal % gxValue);
            errVal *= 2;
        }

        remainderSet.add(seqVal % gxValue);
        return remainderSet.stream().sorted(Comparator.comparingLong(x -> x)).collect(Collectors.toList());
    }

    protected int getGxValue(String gx) {
        return IRREDUCIBLE_POLYNOMIAL_MAP.entrySet().stream()
                .map(Map.Entry::getValue)
                .flatMap(Set::stream)
                .filter(gxPair -> gxPair.getValue().contains(gx))
                .map(Pair::getKey)
                .findFirst().orElse(null);
    }

    public int getSequenceValue() {
        int result = 0;
        for (int i = 0; i < informationalSequence.length(); i++) {
            result = result << 1 + (Character.valueOf('0').equals(informationalSequence.charAt(i)) ? 0 : 1);
        }

        int m = getM();
        while (m > 0) {
            result *= (m % 2) * 2;
            m = m >> 1;
        }

        return result;
    }
}
