class Calculator {

	// Implement your code here
	public double findAverage(int number1,int number2,int number3)
	{
	    double temp=(number1+number2+number3)/3f;
	    return Double.parseDouble(String.format("%.2f",temp));
	}
}

class average {

	public static void main(String args[]) {
		Calculator calculator = new Calculator();
		// Invoke the method findAverage of the Calculator class and display the average
		double x=calculator.findAverage(12,8,15);
		System.out.println(x);
	}
}