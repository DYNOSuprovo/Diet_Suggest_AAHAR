class reqfib{
    public static void main(String[] args) {
        System.out.println(req(5));
}
    public static int req(int i) {
        if(i==0 || i==1)
            return 1;
        int f1=req(i-1);
        int f2=req(i-2);
        int f=f1+f2;
        return f;
    }
}