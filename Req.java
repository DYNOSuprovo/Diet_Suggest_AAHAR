class Req {
    public static void main(String[] args) {
        ReqR(10);
    }
    public static void ReqR(int i) {
            System.out.println(i);
            if(i > 0)
                ReqR(i-1);
            System.out.println(i);
    }
}
