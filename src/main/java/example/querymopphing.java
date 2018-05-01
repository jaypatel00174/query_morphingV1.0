package example;


import de.lmu.ifi.dbs.elki.algorithm.AbstractAlgorithm;
import de.lmu.ifi.dbs.elki.algorithm.clustering.subspace.SubspaceClusteringAlgorithm;
import de.lmu.ifi.dbs.elki.algorithm.clustering.subspace.clique.CLIQUEInterval;
import de.lmu.ifi.dbs.elki.algorithm.clustering.subspace.clique.CLIQUESubspace;
import de.lmu.ifi.dbs.elki.algorithm.clustering.subspace.clique.CLIQUEUnit;
import de.lmu.ifi.dbs.elki.data.Cluster;
import de.lmu.ifi.dbs.elki.data.Clustering;
import de.lmu.ifi.dbs.elki.data.NumberVector;
import de.lmu.ifi.dbs.elki.data.Subspace;
import de.lmu.ifi.dbs.elki.data.model.SubspaceModel;
import de.lmu.ifi.dbs.elki.data.type.TypeInformation;
import de.lmu.ifi.dbs.elki.data.type.TypeUtil;
import de.lmu.ifi.dbs.elki.database.ids.DBIDIter;
import de.lmu.ifi.dbs.elki.database.ids.ModifiableDBIDs;
import de.lmu.ifi.dbs.elki.database.relation.Relation;
import de.lmu.ifi.dbs.elki.database.relation.RelationUtil;
import de.lmu.ifi.dbs.elki.logging.Logging;
import de.lmu.ifi.dbs.elki.math.linearalgebra.Centroid;
import de.lmu.ifi.dbs.elki.math.linearalgebra.Matrix;
import de.lmu.ifi.dbs.elki.utilities.FormatUtil;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.AbstractParameterizer;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.OptionID;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.constraints.CommonConstraints;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.parameterization.Parameterization;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.parameters.*;
import de.lmu.ifi.dbs.elki.utilities.pairs.Pair;
import de.lmu.ifi.dbs.elki.utilities.optionhandling.parameters.FileParameter;

import java.io.*;
import java.util.*;

public class querymopphing<V extends NumberVector>
        extends AbstractAlgorithm<Clustering<SubspaceModel>> implements SubspaceClusteringAlgorithm<SubspaceModel>{

    //Class logger that will be used for error reporting.
    private static final Logging LOG = Logging.getLogger(ExampleAlgorithm.class);

    /**
     * Parameter to specify the number of intervals (units) in each dimension,
     * must be an integer greater than 0.
     * <p>
     * Key: {@code -querymorphing.xsi}
     * </p>
     */
    public static final OptionID XSI_ID = new OptionID("querymorphing.xsi", "The number of intervals (units) in each dimension.");

    /**
     * Parameter to specify the density threshold for the selectivity of a unit,
     * where the selectivity is the fraction of total feature vectors contained in
     * this unit, must be a double greater than 0 and less than 1.
     * <p>
     * Key: {@code -querymorphing.tau}
     * </p>
     */
    public static final OptionID TAU_ID = new OptionID("querymorphing.tau", "The density threshold for the selectivity of a unit, where the selectivity is" + "the fraction of total feature vectors contained in this unit.");

    /**
     * Flag to indicate that only subspaces with large coverage (i.e. the fraction
     * of the database that is covered by the dense units) are selected, the rest
     * will be pruned.
     * <p>
     * Key: {@code -querymorphing.prune}
     * </p>
     */
    public static final OptionID PRUNE_ID = new OptionID("querymorphing.prune", "Flag to indicate that only subspaces with large coverage " + "(i.e. the fraction of the database that is covered by the dense units) " + "are selected, the rest will be pruned.");

    /**
     * String to indicate file name containing initial result set
     * <p>
     * Key: {@code -querymorphing.prune}
     * </p>
     */
    //public static final OptionID RESULTS = new OptionID("querymorphing.results","String to indicate file name containing initial result set");

    public static final OptionID RESULT_FILE_ID = new OptionID("externalresults.file", "The file listing the results.");


    private int xsi;        //Hold the value of {@link #XSI_ID}
    private double tau;     //Hold the value of {@link #TAU_ID}
    private boolean prune;   //Hold the value of {@link #PRUN_ID}
    //private String results; //Hold the value of {@link #RESULT_ID}
    File file;

    public querymopphing(int xsi,double tau,boolean prune, File file) {
        this.xsi = xsi;
        this.tau = tau;
        this.prune = prune;
        this.file = file;
    }

    public Clustering<SubspaceModel> run(Relation<V> relation) {

        StringBuffer buf = new StringBuffer();
        buf.append("This algorithm does not do anything useful yet, except computing mean and variances:");
        buf.append("\n The xsi value is ").append(xsi);
        buf.append("\n The tau value is ").append(tau);
        buf.append("\n The prune value is ").append(prune);
        buf.append("\n The result file name ").append(file);
        LOG.warning(buf);



        // 1. Identification of subspaces that contain clusters
        // TODO: use step logging.
        if(LOG.isVerbose()) {
            LOG.verbose("*** 1. Identification of subspaces that contain clusters ***");
        }

        SortedMap<Integer, List<CLIQUESubspace<V>>> dimensionToDenseSubspaces = new TreeMap<Integer, List<CLIQUESubspace<V>>>();
        List<CLIQUESubspace<V>> denseSubspaces = findOneDimensionalDenseSubspaces(relation);
        dimensionToDenseSubspaces.put(Integer.valueOf(0), denseSubspaces);
        if(LOG.isVerbose()) {
            LOG.verbose("    1-dimensional dense subspaces: " + denseSubspaces.size());
        }

        if(LOG.isDebugging()) {
            for(CLIQUESubspace<V> s : denseSubspaces) {
                LOG.debug(s.toString("      "));
            }
        }

        int dimensionality = RelationUtil.dimensionality(relation);
        for(int k = 2; k <= dimensionality && !denseSubspaces.isEmpty(); k++) {
            denseSubspaces = findDenseSubspaces(relation, denseSubspaces);
            dimensionToDenseSubspaces.put(Integer.valueOf(k - 1), denseSubspaces);
            if(LOG.isVerbose()) {
                LOG.verbose("    " + k + "-dimensional dense subspaces: " + denseSubspaces.size());
            }
            if(LOG.isDebugging()) {
                for(CLIQUESubspace<V> s : denseSubspaces) {
                    LOG.debug(s.toString("      "));
                }
            }
        }

        // 2. Identification of clusters
        if(LOG.isVerbose()) {
            LOG.verbose("*** 2. Identification of clusters ***");
        }
        // build result
        int numClusters = 1;
        Clustering<SubspaceModel> result = new Clustering<SubspaceModel>("CLIQUE clustering", "clique-clustering");
        for(Integer dim : dimensionToDenseSubspaces.keySet()) {
            List<CLIQUESubspace<V>> subspaces = dimensionToDenseSubspaces.get(dim);
            List<Pair<Subspace, ModifiableDBIDs>> modelsAndClusters = determineClusters(subspaces);

            if(LOG.isVerbose()) {
                LOG.verbose("    " + (dim + 1) + "-dimensional clusters: " + modelsAndClusters.size());
            }

            for(Pair<Subspace, ModifiableDBIDs> modelAndCluster : modelsAndClusters) {
                Cluster<SubspaceModel> newCluster = new Cluster<SubspaceModel>(modelAndCluster.second);
                newCluster.setModel(new SubspaceModel(modelAndCluster.first, Centroid.make(relation, modelAndCluster.second)));
                newCluster.setName("cluster_" + numClusters++);
                result.addToplevelCluster(newCluster);
            }
        }

        return result;
    }




    /**
     * Determines the clusters in the specified dense subspaces.
     *
     * @param denseSubspaces the dense subspaces in reverse order by their
     *        coverage
     * @return the clusters in the specified dense subspaces and the corresponding
     *         cluster models
     */
    private List<Pair<Subspace, ModifiableDBIDs>> determineClusters(List<CLIQUESubspace<V>> denseSubspaces) {
        List<Pair<Subspace, ModifiableDBIDs>> clusters = new ArrayList<Pair<Subspace, ModifiableDBIDs>>();

        for(CLIQUESubspace<V> subspace : denseSubspaces) {
            List<Pair<Subspace, ModifiableDBIDs>> clustersInSubspace = subspace.determineClusters();
            if(LOG.isDebugging()) {
                LOG.debugFine("Subspace " + subspace + " clusters " + clustersInSubspace.size());
            }
            clusters.addAll(clustersInSubspace);
        }
        return clusters;
    }

    /**
     * Determines the one dimensional dense subspaces and performs a pruning if
     * this option is chosen.
     *
     * @param database the database to run the algorithm on
     * @return the one dimensional dense subspaces reverse ordered by their
     *         coverage
     */
    private List<CLIQUESubspace<V>> findOneDimensionalDenseSubspaces(Relation<V> database) {
        List<CLIQUESubspace<V>> denseSubspaceCandidates = findOneDimensionalDenseSubspaceCandidates(database);

        if(prune) {
            return pruneDenseSubspaces(denseSubspaceCandidates);
        }

        return denseSubspaceCandidates;
    }

    /**
     * Determines the {@code k}-dimensional dense subspaces and performs a pruning
     * if this option is chosen.
     *
     * @param database the database to run the algorithm on
     * @param denseSubspaces the {@code (k-1)}-dimensional dense subspaces
     * @return a list of the {@code k}-dimensional dense subspaces sorted in
     *         reverse order by their coverage
     */
    private List<CLIQUESubspace<V>> findDenseSubspaces(Relation<V> database, List<CLIQUESubspace<V>> denseSubspaces) {
        List<CLIQUESubspace<V>> denseSubspaceCandidates = findDenseSubspaceCandidates(database, denseSubspaces);

        if(prune) {
            return pruneDenseSubspaces(denseSubspaceCandidates);
        }

        return denseSubspaceCandidates;
    }

    /**
     * Initializes and returns the one dimensional units.
     *
     * @param database the database to run the algorithm on
     * @return the created one dimensional units
     */
    private Collection<CLIQUEUnit<V>> initOneDimensionalUnits(Relation<V> database) {
        int dimensionality = RelationUtil.dimensionality(database);
        // initialize minima and maxima
        double[] minima = new double[dimensionality];
        double[] maxima = new double[dimensionality];
        for(int d = 0; d < dimensionality; d++) {
            maxima[d] = -Double.MAX_VALUE;
            minima[d] = Double.MAX_VALUE;
        }
        // update minima and maxima
        for(DBIDIter it = database.iterDBIDs(); it.valid(); it.advance()) {
            V featureVector = database.get(it);
            updateMinMax(featureVector, minima, maxima);
        }
        for(int i = 0; i < maxima.length; i++) {
            maxima[i] += 0.0001;
        }

        // determine the unit length in each dimension
        double[] unit_lengths = new double[dimensionality];
        for(int d = 0; d < dimensionality; d++) {
            unit_lengths[d] = (maxima[d] - minima[d]) / xsi;
        }

        if(LOG.isDebuggingFiner()) {
            StringBuilder msg = new StringBuilder();
            msg.append("   minima: ").append(FormatUtil.format(minima, ", ", FormatUtil.NF2));
            msg.append("\n   maxima: ").append(FormatUtil.format(maxima, ", ", FormatUtil.NF2));
            msg.append("\n   unit lengths: ").append(FormatUtil.format(unit_lengths, ", ", FormatUtil.NF2));
            LOG.debugFiner(msg.toString());
        }

        // determine the boundaries of the units
        double[][] unit_bounds = new double[xsi + 1][dimensionality];
        for(int x = 0; x <= xsi; x++) {
            for(int d = 0; d < dimensionality; d++) {
                if(x < xsi) {
                    unit_bounds[x][d] = minima[d] + x * unit_lengths[d];
                }
                else {
                    unit_bounds[x][d] = maxima[d];
                }
            }
        }
        if(LOG.isDebuggingFiner()) {
            StringBuilder msg = new StringBuilder();
            msg.append("   unit bounds ").append(FormatUtil.format(new Matrix(unit_bounds), "   "));
            LOG.debugFiner(msg.toString());
        }

        // build the 1 dimensional units
        List<CLIQUEUnit<V>> units = new ArrayList<CLIQUEUnit<V>>((xsi * dimensionality));
        for(int x = 0; x < xsi; x++) {
            for(int d = 0; d < dimensionality; d++) {
                units.add(new CLIQUEUnit<V>(new CLIQUEInterval(d, unit_bounds[x][d], unit_bounds[x + 1][d])));
            }
        }

        if(LOG.isDebuggingFiner()) {
            StringBuilder msg = new StringBuilder();
            msg.append("   total number of 1-dim units: ").append(units.size());
            LOG.debugFiner(msg.toString());
        }

        return units;
    }

    /**
     * Updates the minima and maxima array according to the specified feature
     * vector.
     *
     * @param featureVector the feature vector
     * @param minima the array of minima
     * @param maxima the array of maxima
     */
    private void updateMinMax(V featureVector, double[] minima, double[] maxima) {
        if(minima.length != featureVector.getDimensionality()) {
            throw new IllegalArgumentException("FeatureVectors differ in length.");
        }
        for(int d = 0; d < featureVector.getDimensionality(); d++) {
            if((featureVector.doubleValue(d)) > maxima[d]) {
                maxima[d] = (featureVector.doubleValue(d));
            }
            if((featureVector.doubleValue(d)) < minima[d]) {
                minima[d] = (featureVector.doubleValue(d));
            }
        }
    }

    /**
     * Determines the one-dimensional dense subspace candidates by making a pass
     * over the database.
     *
     * @param database the database to run the algorithm on
     * @return the one-dimensional dense subspace candidates reverse ordered by
     *         their coverage
     */
    private List<CLIQUESubspace<V>> findOneDimensionalDenseSubspaceCandidates(Relation<V> database) {
        Collection<CLIQUEUnit<V>> units = initOneDimensionalUnits(database);
        // identify dense units
        double total = database.size();
        for(DBIDIter it = database.iterDBIDs(); it.valid(); it.advance()) {
            V featureVector = database.get(it);
            for(CLIQUEUnit<V> unit : units) {
                unit.addFeatureVector(it, featureVector);
            }
        }

        Collection<CLIQUEUnit<V>> denseUnits = new ArrayList<CLIQUEUnit<V>>();
        Map<Integer, CLIQUESubspace<V>> denseSubspaces = new HashMap<Integer, CLIQUESubspace<V>>();
        for(CLIQUEUnit<V> unit : units) {
            // unit is a dense unit
            if(unit.selectivity(total) >= tau) {
                denseUnits.add(unit);
                // add the dense unit to its subspace
                int dim = unit.getIntervals().iterator().next().getDimension();
                CLIQUESubspace<V> subspace_d = denseSubspaces.get(Integer.valueOf(dim));
                if(subspace_d == null) {
                    subspace_d = new CLIQUESubspace<V>(dim);
                    denseSubspaces.put(Integer.valueOf(dim), subspace_d);
                }
                subspace_d.addDenseUnit(unit);
            }
        }

        if(LOG.isDebugging()) {
            StringBuilder msg = new StringBuilder();
            msg.append("   number of 1-dim dense units: ").append(denseUnits.size());
            msg.append("\n   number of 1-dim dense subspace candidates: ").append(denseSubspaces.size());
            LOG.debugFine(msg.toString());
        }

        List<CLIQUESubspace<V>> subspaceCandidates = new ArrayList<CLIQUESubspace<V>>(denseSubspaces.values());
        Collections.sort(subspaceCandidates, new CLIQUESubspace.CoverageComparator());
        return subspaceCandidates;
    }

    /**
     * Determines the {@code k}-dimensional dense subspace candidates from the
     * specified {@code (k-1)}-dimensional dense subspaces.
     *
     * @param database the database to run the algorithm on
     * @param denseSubspaces the {@code (k-1)}-dimensional dense subspaces
     * @return a list of the {@code k}-dimensional dense subspace candidates
     *         reverse ordered by their coverage
     */
    private List<CLIQUESubspace<V>> findDenseSubspaceCandidates(Relation<V> database, List<CLIQUESubspace<V>> denseSubspaces) {
        // sort (k-1)-dimensional dense subspace according to their dimensions
        List<CLIQUESubspace<V>> denseSubspacesByDimensions = new ArrayList<CLIQUESubspace<V>>(denseSubspaces);
        Collections.sort(denseSubspacesByDimensions, new Subspace.DimensionComparator());

        // determine k-dimensional dense subspace candidates
        double all = database.size();
        List<CLIQUESubspace<V>> denseSubspaceCandidates = new ArrayList<CLIQUESubspace<V>>();

        while(!denseSubspacesByDimensions.isEmpty()) {
            CLIQUESubspace<V> s1 = denseSubspacesByDimensions.remove(0);
            for(CLIQUESubspace<V> s2 : denseSubspacesByDimensions) {
                CLIQUESubspace<V> s = s1.join(s2, all, tau);
                if(s != null) {
                    denseSubspaceCandidates.add(s);
                }
            }
        }

        // sort reverse by coverage
        Collections.sort(denseSubspaceCandidates, new CLIQUESubspace.CoverageComparator());
        return denseSubspaceCandidates;
    }

    /**
     * Performs a MDL-based pruning of the specified dense subspaces as described
     * in the CLIQUE algorithm.
     *
     * @param denseSubspaces the subspaces to be pruned sorted in reverse order by
     *        their coverage
     * @return the subspaces which are not pruned reverse ordered by their
     *         coverage
     */
    private List<CLIQUESubspace<V>> pruneDenseSubspaces(List<CLIQUESubspace<V>> denseSubspaces) {
        int[][] means = computeMeans(denseSubspaces);
        double[][] diffs = computeDiffs(denseSubspaces, means[0], means[1]);
        double[] codeLength = new double[denseSubspaces.size()];
        double minCL = Double.MAX_VALUE;
        int min_i = -1;

        for(int i = 0; i < denseSubspaces.size(); i++) {
            int mi = means[0][i];
            int mp = means[1][i];
            double log_mi = mi == 0 ? 0 : StrictMath.log(mi) / StrictMath.log(2);
            double log_mp = mp == 0 ? 0 : StrictMath.log(mp) / StrictMath.log(2);
            double diff_mi = diffs[0][i];
            double diff_mp = diffs[1][i];
            codeLength[i] = log_mi + diff_mi + log_mp + diff_mp;

            if(codeLength[i] <= minCL) {
                minCL = codeLength[i];
                min_i = i;
            }
        }

        return denseSubspaces.subList(0, min_i + 1);
    }

    /**
     * The specified sorted list of dense subspaces is divided into the selected
     * set I and the pruned set P. For each set the mean of the cover fractions is
     * computed.
     *
     * @param denseSubspaces the dense subspaces in reverse order by their
     *        coverage
     * @return the mean of the cover fractions, the first value is the mean of the
     *         selected set I, the second value is the mean of the pruned set P.
     */
    private int[][] computeMeans(List<CLIQUESubspace<V>> denseSubspaces) {
        int n = denseSubspaces.size() - 1;

        int[] mi = new int[n + 1];
        int[] mp = new int[n + 1];

        double resultMI = 0;
        double resultMP = 0;

        for(int i = 0; i < denseSubspaces.size(); i++) {
            resultMI += denseSubspaces.get(i).getCoverage();
            resultMP += denseSubspaces.get(n - i).getCoverage();
            mi[i] = (int) Math.ceil(resultMI / (i + 1));
            if(i != n) {
                mp[n - 1 - i] = (int) Math.ceil(resultMP / (i + 1));
            }
        }

        int[][] result = new int[2][];
        result[0] = mi;
        result[1] = mp;

        return result;
    }

    /**
     * The specified sorted list of dense subspaces is divided into the selected
     * set I and the pruned set P. For each set the difference from the specified
     * mean values is computed.
     *
     * @param denseSubspaces denseSubspaces the dense subspaces in reverse order
     *        by their coverage
     * @param mi the mean of the selected sets I
     * @param mp the mean of the pruned sets P
     * @return the difference from the specified mean values, the first value is
     *         the difference from the mean of the selected set I, the second
     *         value is the difference from the mean of the pruned set P.
     */
    private double[][] computeDiffs(List<CLIQUESubspace<V>> denseSubspaces, int[] mi, int[] mp) {
        int n = denseSubspaces.size() - 1;

        double[] diff_mi = new double[n + 1];
        double[] diff_mp = new double[n + 1];

        double resultMI = 0;
        double resultMP = 0;

        for(int i = 0; i < denseSubspaces.size(); i++) {
            double diffMI = Math.abs(denseSubspaces.get(i).getCoverage() - mi[i]);
            resultMI += diffMI == 0.0 ? 0 : StrictMath.log(diffMI) / StrictMath.log(2);
            double diffMP = (i != n) ? Math.abs(denseSubspaces.get(n - i).getCoverage() - mp[n - 1 - i]) : 0;
            resultMP += diffMP == 0.0 ? 0 : StrictMath.log(diffMP) / StrictMath.log(2);
            diff_mi[i] = resultMI;
            if(i != n) {
                diff_mp[n - 1 - i] = resultMP;
            }
        }
        double[][] result = new double[2][];
        result[0] = diff_mi;
        result[1] = diff_mp;

        return result;
    }
    
    @Override
    public TypeInformation[] getInputTypeRestriction() {
        return TypeUtil.array(TypeUtil.NUMBER_VECTOR_FIELD);
    }


    public static class Parameterizer<V extends NumberVector> extends AbstractParameterizer {
        protected int xsi;

        protected double tau;

        protected boolean prune;

        protected String results;

        File file;


        @Override
        protected void makeOptions(Parameterization config) {
            super.makeOptions(config);



            IntParameter xsiP = new IntParameter(XSI_ID);
            xsiP.addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT);
            if (config.grab(xsiP)) {
                xsi = xsiP.intValue();
            }

            DoubleParameter tauP = new DoubleParameter(TAU_ID);
            tauP.addConstraint(CommonConstraints.GREATER_THAN_ZERO_DOUBLE);
            tauP.addConstraint(CommonConstraints.LESS_THAN_ONE_DOUBLE);
            if (config.grab(tauP)) {
                tau = tauP.doubleValue();
            }

            Flag pruneF = new Flag(PRUNE_ID);
            if (config.grab(pruneF)) {
                prune = pruneF.isTrue();
            }


            file = getParameterResultFile(config);
            /**StringParameter strP = new StringParameter(RESULTS);
            if(config.grab(strP)){
                results = strP.getValue();
            }**/

        }


        protected static File getParameterResultFile(Parameterization config) {
            final FileParameter param = new FileParameter(RESULT_FILE_ID, FileParameter.FileType.INPUT_FILE);
            if(config.grab(param)) {
                return param.getValue();
            }
            return null;
        }

        @Override
        protected querymopphing<V> makeInstance() {
            return new querymopphing<>(xsi, tau, prune,file);
        }

    }

    @Override
    protected Logging getLogger() {
        return LOG;
    }
}
