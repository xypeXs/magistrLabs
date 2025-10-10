package edu.rsatu.encryption;

import edu.rsatu.encryption.dto.DecodeResult;
import edu.rsatu.encryption.dto.Pair;
import edu.rsatu.encryption.utils.NumberUtils;

import java.util.ArrayList;
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
            5, Set.of(Pair.of(37, "x⁵ + x² + 1"), Pair.of(41, "x⁵ + x³ + 1"), Pair.of(47, "x⁵ + x³ + x² + x + 1"), Pair.of(55, "x⁵ + x⁴ + x² + x + 1"), Pair.of(59, "x⁵ + x⁴ + x³ + x + 1"), Pair.of(61, "x⁵ + x⁴ + x³ + x² + 1")) // 5 - 100101, 101001, 101111, 110111, 111011, 111101
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

    // 2^m-1 >= n -> 2^m - m - 1 >= k
    private void initM() {
        int k = getK();
        int powerOfM = 2;
        int mIterator = 1;
        for (; mIterator < k && (powerOfM - mIterator - 1 < k); mIterator++) {
            powerOfM *= 2;
        }

        this.m = mIterator;
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

    public int encode(String gx) {
        Integer seqVal = NumberUtils.getBinaryNumber(informationalSequence);
        Integer finalSeqVal = seqVal;
        List<Integer> checkBitsList = getGT(gx).stream().map(x -> NumberUtils.bitByBitAndThenXor(x, finalSeqVal)).toList();

        for (Integer checkBit : checkBitsList) {
            seqVal = (seqVal << 1) + checkBit;
        }

        return (seqVal << 1) + NumberUtils.getXorOfAllBits(seqVal);
    }

    public DecodeResult decode(int sequenceToDecode, String gx) {
        int parityBit = NumberUtils.getXorOfAllBits(sequenceToDecode);
        int seqToDecode = sequenceToDecode >> 1;
        int gxBin = getGxValue(gx);
        if (NumberUtils.modPolynomial(seqToDecode, gxBin) == 0) {
            if (parityBit == 0) {
                return new DecodeResult(0, 0, 0, 0, seqToDecode >> getM());
            }

            if (parityBit == 1) {
                return new DecodeResult(0, 0, 1, -1, seqToDecode >> getM());
            }
        }

        List<Integer> H = getH(gx);
        int s = NumberUtils.getNumberFromBinaryList(getHT(gx).stream().map(hi -> NumberUtils.bitByBitAndThenXor(seqToDecode, hi)).toList());
        int hPos = H.indexOf(s);
        int n = 0, eps = 0, r = 0, decodedSeq = 0;
        if (hPos != -1 && parityBit == 0) {
            eps = 1 << (getN() - hPos - 1);
            return new DecodeResult(s, eps, 2, 0, seqToDecode >> getM());
        }

        if (hPos != -1 && parityBit == 1) {
            n = getN() - hPos - 1;
            eps = 1 << n;
            r = 1;

            decodedSeq = seqToDecode ^ eps;
            return new DecodeResult(s, eps, r, n, decodedSeq >> getM());
        }

        return new DecodeResult(0, 0, 2, 0, 0);
    }

    public List<Integer> getHT(String gx) {
        return NumberUtils.transposeBinaryMatr(getH(gx));
    }

    public List<Integer> getH(String gx) {
        List<Integer> remainderList = getG(gx);
        int e = 1 << (getM() - 1);

        for (int i = 0; i < getM(); i++) {
            remainderList.add(e);
            e >>= 1;
        }

        return remainderList;
    }

    public List<Integer> getGT(String gx) {
        return NumberUtils.transposeBinaryMatr(getG(gx));
    }

    public List<Integer> getG(String gx) {
        return getRemainderList(gx).stream().limit(getK()).collect(Collectors.toList());
    }

    public List<Integer> getRemainderList(String gx) {
        int gxValue = getGxValue(gx);
        int errVal = 1 << (getN() - 1);
        List<Integer> remainderList = new ArrayList<>(getN());
        for (int i = 0; i < getN(); i++) {
            remainderList.add(NumberUtils.modPolynomial(errVal, gxValue));
            errVal = errVal >> 1;
        }

        return remainderList.stream().distinct().collect(Collectors.toList());
    }

    protected int getGxValue(String gx) {
        return IRREDUCIBLE_POLYNOMIAL_MAP.entrySet().stream()
                .map(Map.Entry::getValue)
                .flatMap(Set::stream)
                .filter(gxPair -> gxPair.getValue().equals(gx))
                .map(Pair::getKey)
                .findFirst().orElse(null);
    }
}
