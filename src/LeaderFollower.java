import java.io.*;
import java.util.*;

public class LeaderFollower extends ClusteringAlgorithm
{
	// Intradistance of clusters
	private double maxDis;
	
	// Dimensionality of the vectors
	private int dim;
	
	// Threshold above which the corresponding html is prefetched
	private double prefetchThreshold;
	
	// learning rate
	private double alpha;
	
	// This class represents the clusters, it contains the prototype (the mean of all it's members)
	// and a memberlist with the ID's (Integer objects) of the datapoints that are member of that cluster.  
	private Vector<Cluster> clusters;

	static class Cluster
	{
		float[] prototype;

		Set<Integer> currentMembers;

		public Cluster()
		{
			currentMembers = new HashSet<Integer>();
		}
	}
	
	// Vector which contains the train/test data
	private Vector<float[]> trainData;
	private Vector<float[]> testData;

	// Results of test()
	private double hitrate;
	private double accuracy;
	
	public LeaderFollower(double maxDis, Vector<float[]> trainData, Vector<float[]> testData, int dim)
	{
		this.maxDis = maxDis;
		prefetchThreshold = 0.5;
		alpha = 0.1;
		
		this.trainData = trainData;
		this.testData = testData; 
		this.dim = dim;
		
		// Now we put the clusters in a vector because we don't know in advance how many clusters there will be.
		clusters = new Vector<Cluster>();
	}

	public boolean train()
	{
		// classify for each trainDataPoint which cluster prototype is its NN,
		// if the minimal distance < maxDis make it member of the cluster
		// or else make a new cluster.
		return true;
	}

	public boolean test()
	{
		// iterate along all clients. Assumption: the same clients are in the same order as in the testData
		// for each client find the cluster of which it is a member
		// get the actual testData (the vector) of this client
		// iterate along all dimensions
		// and count prefetched htmls
		// count number of hits
		// count number of requests
		// set the global variables hitrate and accuracy to their appropriate value
		return true;
	}


	// The following members are called by runClustering, in order to present information to the user
	public void showTest()
	{
		System.out.println("Prefetch threshold = " + prefetchThreshold);
		System.out.println("Alpha=" + alpha);       
		System.out.println("Hitrate: " + hitrate);
		System.out.println("Accuracy: " + accuracy);
		System.out.println("Hitrate+Accuracy=" + (hitrate + accuracy));
	}
	
	public void showMembers()
	{
		for (int i = 0; i < clusters.size(); i++)
			System.out.println("\nMembers cluster["+i+"] :" + clusters.get(i).currentMembers);
	}
	 
	public void showPrototypes() 
	{
		for (int i = 0; i < clusters.size(); i++) {
			System.out.print("\nPrototype cluster["+i+"] :");
			
			float[] prototype = clusters.get(i).prototype;

			for (int i2=0; i2 < dim; i2++)
				System.out.print(prototype[i2]+" ");
			
			System.out.println();
		}
	}

	// with this function you can set the prefetch threshold.
	public void setPrefetchThreshold(double prefetchThreshold)
	{
		this.prefetchThreshold = prefetchThreshold;
	}
}
