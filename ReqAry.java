class ReqAry{
    public static void main(String[] args) {
        int a[]={1,3,3,4,9};
        if (sorted(a, 0)) {
            System.out.println("Array is sorted in ascending order.");
        } else {
            System.out.println("Array is not sorted.");
        }
    }
    public static boolean sorted(int[] arr,int i){
            if(i==arr.length-1){
                return true;
            }
            if(arr[i]>arr[i+1]){
                return false;
            }
            return sorted(arr,i+1);
        }
}