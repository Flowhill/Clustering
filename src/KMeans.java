import java.util.*;

public class KMeans extends ClusteringAlgorithm
{
	// Number of clusters
	private int k;

	// Dimensionality of the vectors
	private int dim;
	
	// Threshold above which the corresponding html is prefetched
	private double prefetchThreshold;
	
	// Array of k clusters, class cluster is used for easy bookkeeping
	private Cluster[] clusters;
	
	// This class represents the clusters, it contains the prototype (the mean of all it's members)
	// and memberlists with the ID's (which are Integer objects) of the datapoints that are member of that cluster.
	// You also want to remember the previous members so you can check if the clusters are stable.
	static class Cluster
	{
		float[] prototype;

		Set<Integer> currentMembers;
		Set<Integer> previousMembers;
		  
		public Cluster()
		{
			currentMembers = new HashSet<>();
			previousMembers = new HashSet<>();
		}
	}
	// These vectors contains the feature vectors you need; the feature vectors are float arrays.
	// Remember that you have to cast them first, since vectors return objects.
	private Vector<float[]> trainData;
	private Vector<float[]> testData;

	// Results of test()
	private double hitrate;
	private double accuracy;

	public int isIdenticalSet(Set<Integer> h1, Set<Integer> h2) {
		/// Source: http://stackoverflow.com/questions/11888554/way-to-check-if-two-collections-contain-the-same-elements-independent-of-order
		/// Altered to work for sets and return 1 if the sets are equal, 0 if they are not
		if ( h1.size() != h2.size() ) {
			return 0;
		}
		Set<Integer> clone = new HashSet<>(h2); // just use h2 if you don't need to save the original h2
		for (Object aH1 : h1) {
			int remover = (int) aH1;
			if (clone.contains(remover)) { // replace clone with h2 if not concerned with saving data from h2
				clone.remove(remover);
			} else {
				return 0;
			}
		}
		return 1; // will only return true(1) if sets are equal
	}
	
	public KMeans(int k, Vector<float[]> trainData, Vector<float[]> testData, int dim)
	{
		this.k = k;
		this.trainData = trainData;
		this.testData = testData; 
		this.dim = dim;
		prefetchThreshold = 0.5;
		
		// Here k new cluster are initialized
		clusters = new Cluster[k];
		for (int ic = 0; ic < k; ic++)
			clusters[ic] = new Cluster();
	}

	public float[] computePrototype(int clusterNumber, float[] prior){
		float[] prototype = new float[200];
		for(int i : clusters[clusterNumber].currentMembers) {
			for (int h = 0; h < 200; h++) {
				prototype[h] += trainData.get(i)[h];
			}
		}
		int size = clusters[clusterNumber].currentMembers.size();
		for (int h = 0; h < 200; h++) {
			prototype[h] += 2*prior[h];
			prototype[h] /= size + 2;
		}
		return prototype;
	}


	public boolean train()
	{
	 	//implement k-means algorithm here:
		// Step 1: Select an initial random partitioning with k clusters
		// Step 2: Generate a new partition by assigning each datapoint to its closest cluster center
		// Step 3: recalculate cluster centers
		// Step 4: repeat until clustermembership stabilizes

		/// Step 1
		Random rand = new Random();
		for(int i = 0; i < trainData.size(); i++ ){
			int randValue = rand.nextInt(k); ///Generates a random value from 0 to k-1
			clusters[randValue].currentMembers.add(i); ///Add index to currentMembers
		}

		/// Compute Priors:
		float[] prior = new float[200];
		for (int h = 0; h < 200; h++) {
			prior[h] = 1;
			int size = trainData.size();
			for (float[] aTrainData : trainData) {
				prior[h] += aTrainData[h];
			}
			prior[h] /= size + 2;
		}

		int stabilized = 0; /// Assume the situation is unstable (0)
		while(stabilized == 0) { /// Step 4: Repeat k-means while the situation is unstable
			/// Step 3
			for (int j = 0; j < k; j++) {
				clusters[j].prototype = computePrototype( j, prior);
				clusters[j].previousMembers = clusters[j].currentMembers;
				clusters[j].currentMembers.clear();
			}
			/// Step 2
			for (int i = 0; i < trainData.size(); i++) {   // loop over all clients
				int best = -1;
				double minDistance = Double.POSITIVE_INFINITY; //distance = infinity at first
				for (int j = 0; j < k; j++) {   // loop over clusters to find the closest centroid
					double distance = 0;
					float[] x = clusters[j].prototype;
					float[] y = trainData.get(i);
					for (int h = 0; h < 200; h++) {
						distance += Math.pow(x[h] - y[h], 2);
					}
					distance = Math.sqrt(distance);
					if (distance < minDistance) {
						minDistance = distance;
						best = j;
					}
				}
				clusters[best].currentMembers.add(i);

			}
			stabilized = 1; /// Assume the situation is stable (1)
			for (int j = 0; j < k; j++) { /// Check whether the situation is truly stable (will become 0 when unstable, 1 if stable)
				stabilized *= isIdenticalSet(clusters[j].currentMembers, clusters[j].previousMembers); /// If at any point 0 is returned in the loop -> stabilized = 0
			}
		}
		return false;
	}

	public boolean test()
	{
		int  prefetches,hits,requests;
		prefetches =hits =requests =0;
		for (int i = 0; i < testData.size(); i++) {    // iterate along all clients. Assumption: the same clients are in the same order as in the testData
			int  clusterNumber = - 1 ;
			for (int j = 0; j < k; j++) {                // for each client find the cluster of which it is a member
				if (clusters[j].currentMembers.contains(i)) {
					clusterNumber = j;
				}
			}
			float[] prototype = clusters[clusterNumber].prototype;
			float[] datapoint = testData.get(i); 	// get the actual testData (the vector) of this client
			boolean a, b;
			for (int h = 0; h < 200; h++) { 		// iterate along all dimensions
				a = datapoint[h] == 1;
				b = prototype[h] > prefetchThreshold;
				if (b)	 prefetches++; // and count prefetched htmls
				if (a & b)     hits++; // count number of hits
				if (a)	   requests++; // count number of requests
			}
		}
		// set the global variables hitrate and accuracy to their appropriate value:
		this.hitrate= (double)hits / (double)requests;
		this.accuracy=(double)hits/ (double)prefetches;

		return true;
	}


	// The following members are called by RunClustering, in order to present information to the user
	public void showTest()
	{
		System.out.println("Prefetch threshold=" + this.prefetchThreshold);
		System.out.println("Hitrate: " + this.hitrate);
		System.out.println("Accuracy: " + this.accuracy);
		System.out.println("Hitrate+Accuracy=" + (this.hitrate + this.accuracy));
	}
	
	public void showMembers()
	{
		for (int i = 0; i < k; i++)
			System.out.println("\nMembers cluster["+i+"] :" + clusters[i].currentMembers);
	}
	
	public void showPrototypes()
	{
		for (int ic = 0; ic < k; ic++) {
			System.out.print("\nPrototype cluster["+ic+"] :");
			
			for (int ip = 0; ip < dim; ip++)
				System.out.print(clusters[ic].prototype[ip] + " ");
			
			System.out.println();
		}
	}

	// With this function you can set the prefetch threshold.
	public void setPrefetchThreshold(double prefetchThreshold)
	{
		this.prefetchThreshold = prefetchThreshold;
	}
}
