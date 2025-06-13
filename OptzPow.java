class OptzPow{
    public static void main(String[] args) {
        int a=10,b=3;
        System.out.println(power(a,b));    
    }
    public static int power(int a,int b){
        if(b==0){
            return 1;
        }
        int p=power(a,b/2)*power(a, b/2);
        if(b%2!=0){
            p=a*p;
        }
        return p;
    }
}