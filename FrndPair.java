class FrndPair{
    public static void main(String[] args){
        System.out.println(friend(5));
    }
    public static int friend(int n){
        if(n==1 || n==2)
            return n;
        //choice
        //single
        int f=friend(n-1);
        //pair
        int f2=friend(n-2);
        int pair=(n-1)*f2;
        //total
        int total= f+pair;
        return total;
    }
}