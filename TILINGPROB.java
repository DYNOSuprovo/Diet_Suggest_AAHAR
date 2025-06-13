class TILINGPROB {
    public static int tilling(int n) {
        if (n == 0 || n == 1) {
            return 1;
        }
        int fnm1 = tilling(n - 1);
        int fnm2 = tilling(n - 2); 
        int toWays = fnm1 + fnm2;
        return toWays;
    }
    public static void main(String[] args) {
        System.out.println(tilling(2));
        System.out.println(tilling(3));
        System.out.println(tilling(4));
    }
}