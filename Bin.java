class Bin{
    public static void main(String[] args) {
        BinStr(3,0,"");
    }
    public static void BinStr(int n,int l,String str){
        if(n==0){
            System.err.println(str);
            return;
        }
        BinStr(n-1, 0, str+"0");
        if(l==0)
            BinStr(n-1, 1, str+"1");
    }
}
