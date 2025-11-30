package edu.rsatu.encryption.dto;

public class DecodeResult {

    private int s;
    private int eps;
    private int r;
    private int n;
    private int decodedSeq;

    public DecodeResult(int s, int eps, int r, int n, int decodedSeq) {
        this.s = s;
        this.eps = eps;
        this.r = r;
        this.n = n;
        this.decodedSeq = decodedSeq;
    }

    public int getS() {
        return s;
    }

    public void setS(int s) {
        this.s = s;
    }

    public int getEps() {
        return eps;
    }

    public void setEps(int eps) {
        this.eps = eps;
    }

    public int getR() {
        return r;
    }

    public void setR(int r) {
        this.r = r;
    }

    public int getN() {
        return n;
    }

    public void setN(int n) {
        this.n = n;
    }

    public int getDecodedSeq() {
        return decodedSeq;
    }

    public void setDecodedSeq(int decodedSeq) {
        this.decodedSeq = decodedSeq;
    }
}
