/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.utils.stats

import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix, SparseMatrix, Vector => OldVector}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD
import com.twitter.algebird.Monoid._
import com.twitter.algebird.Operators._

object OpStatistics {

  /**
   * Two-element result tuple containing a map of labels to values which is used for eg. pointwise mutual information
   * or the contingency matrix itself.
   */
  object LabelWiseValues {
    type Type = Map[String, Array[Double]]
    def empty: Type = Map.empty
  }

  /**
   * Converts a matrix object into a map from column index -> Array of column values
   *
   * @param arr   Input matrix (can be dense or sparse)
   * @return      Map from column index -> Array of column values
   */
  private def matrixToLableWiseStats(arr: Matrix): LabelWiseValues.Type = {
    arr.colIter.zipWithIndex.map {
      case (vec, idx) => idx.toString -> vec.toArray
    }.toMap
  }

  /**
   * Assumes that we have already computed a MultivariateStatisticsSummary on the RDD, so we can use that info here.
   * This defines an RDD aggregation that calculates all the correlations with the label. Data is assumed to be laid
   * out in an RDD[org.apache.spark.mllib.linalg.Vector] where the label is the last element.
   *
   * @param featuresAndLabel Input RDD consisting of a single array containing the feature vector with the label as
   *                         the last element
   * @return  Array of correlations of each feature vector element with the label
   */
  def computeCorrelationsWithLabel(
    featuresAndLabel: RDD[OldVector],
    colStats: MultivariateStatisticalSummary,
    numOfRows: Long
  ): Array[Double] = {
    require(numOfRows > 1, s"computeCorrelationsWithLabel called on matrix with only $numOfRows rows." +
      "  Cannot compute the covariance of a matrix with <= 1 row.")
    val means = colStats.mean.toArray
    val variances = colStats.variance.toArray
    val stdDevs = colStats.variance.toArray.map(math.sqrt)
    val numOfCols = means.size

    // TODO: Look into compensated summation algorithm for use when nRows is large and we need to worry about roundoff

    // First compute the covariance of each feature column with the label column
    val covariancesWithLabel = featuresAndLabel.treeAggregate(zeroValue = new Array[Double](numOfCols))(
      seqOp = (agg, el) => agg +
        (0 until numOfCols).map(i => (el(i) - means(i)) * (el(numOfCols - 1) - means.last)).toArray,
      combOp = (x1, x2) => x1 + x2
    )

    val correlationsWithLabel = (0 until numOfCols).map(i =>
      covariancesWithLabel(i)/(stdDevs(i) * stdDevs.last * (numOfRows - 1))
    ).toArray

    correlationsWithLabel
  }

  /**
   * Container for association rule confidence and supports
   *
   * @param maxConfidences  Array of maximum confidence values, one per contingency matrix row
   * @param supports        Array of support values for each categorical value, one per contingency matrix row
   */
  case class ConfidenceResults(maxConfidences: Array[Double], supports: Array[Double])

  /**
   * Container class for statistics calculated from contingency matrices constructed from categorical variables
   *
   * @param chiSquaredResults   Chi-squared test results for the given contingency matrix
   * @param pointwiseMutualInfo Map between feature name in feature vector and map of pointwise mutual information
   *                            values between that feature and all values the label can take
   * @param contingencyMatrix   Actual (unfiltered) contingency matrix that the rest of the results are calculated from
   * @param mutualInfo          Map between feature name in feature vector and the mutual information with the label
   * @param confidenceResults   Association rule details (confidences + supports)
   */
  case class ContingencyStats
  (
    chiSquaredResults: ChiSquaredResults,
    pointwiseMutualInfo: LabelWiseValues.Type,
    contingencyMatrix: LabelWiseValues.Type,
    mutualInfo: Double,
    confidenceResults: ConfidenceResults
  )

  /**
   * Case class for holding results of the Chi-squared statistical test we use for calculating Cramer's V
   *
   * @param cramersV        Cramer's V value
   * @param chiSquaredStat  Actual Chi-squared statistic
   * @param pValue          P-value
   */
  case class ChiSquaredResults(cramersV: Double, chiSquaredStat: Double, pValue: Double)

  /**
   * Filters out empty rows and columns from the contingency matrix
   *
   * @param contingency   Input contingency matrix (each column represents a label, each row a feature value)
   * @return              Contingency matrix with any row or col consisting entirely of zeroes removed
   */
  private def filterEmpties(contingency: Matrix): Matrix = {
    if (contingency.numActives <= 0) return contingency

    // First check contingency matrix to see make sure we can compute a chi-squared test on it
    val (rows, cols) = (contingency.numRows, contingency.numCols)
    // Add all the columns together
    val summedCols = contingency.multiply(new DenseMatrix(cols, 1, Array.fill(cols)(1.0)))

    // The matrix has a column for each label, and a row for each indicator value for the specific indicator group.
    // Since for any topK vectorization, if K < the actual number of choices in your categorical variable then it
    // will still have an "other" category created, but it will always be empty. We can either check this for the
    // other column specifically, or allow empty rows (and maybe columns) to be removed if there's still enough matrix
    // to compute a chi-squared statistic on it.

    // summedCols is a single column matrix so can find rows by looking at the array indices directly
    val emptyRowIndices: Array[Int] = summedCols.values.zipWithIndex.filter(_._1 == 0.0).map(_._2)
    // DenseMatrix is column-major so filter rows by transposing and iterating through columns to get array of values
    val filteredRows =
      contingency.transpose.colIter.zipWithIndex.filterNot(f => emptyRowIndices.contains(f._2)).map(_._1.toArray)
    val filteredRowMatrix =
      new DenseMatrix(cols, rows - emptyRowIndices.length, filteredRows.flatten.toArray[Double]).transpose

    // Now check if there are any columns of all zeros by summing all the rows together
    val summedRows = {
      new DenseMatrix(1, filteredRowMatrix.numRows, Array.fill(filteredRowMatrix.numRows)(1.0))
        .multiply(filteredRowMatrix)
    }
    val emptyColIndices: Array[Int] = summedRows.values.zipWithIndex.filter(_._1 == 0.0).map(_._2)
    val filteredCols =
      filteredRowMatrix.colIter.zipWithIndex
        .filterNot(f => emptyColIndices.contains(f._2))
        .flatMap(_._1.toArray).toArray[Double]

    new DenseMatrix(rows - emptyRowIndices.length, cols - emptyColIndices.length, filteredCols)
  }

  /**
   * Compute Cramer's V statistic (and other goodies from the chi-squared test) based on the given contingency matrix
   * and sample size. This will filter out the empty rows/cols from the contingency matrix before calculating
   * Cramer's V since the chi-squared statistic used inside the Cramer's V calculation cannot be done on matrices
   * with any rows/cols of all zeros
   *
   * @see https://goo.gl/Fw6ZMN
   *
   * @param contingency Contingency matrix
   * @return ChiSquaredResults object containing results for that contingency matrix
   */
  private[stats] def chiSquaredTest(contingency: Matrix): ChiSquaredResults = {
    val filteredMatrix = filterEmpties(contingency)
    chiSquaredTestOnFiltered(filteredMatrix)
  }

  /**
   * Cramer's V calculation, assuming the matrix passed in has had all its empty rows and columns filtered out. It
   * will return Double.NaN if the contingency matrix does not have 2 or more rows/cols (each). Note
   * that unlike the default behavior of scipy.stats.chi2_contingency, the chi-squared test here does not apply any
   * corrections to the contingency matrix (eg. Yates').
   *
   * @param filteredMatrix Contingency matrix with no rows/cols of zeros
   * @return ChiSquaredResults object containing results for that contingency matrix
   */
  private[stats] def chiSquaredTestOnFiltered(filteredMatrix: Matrix): ChiSquaredResults = {
    if (filteredMatrix.numRows > 1 && filteredMatrix.numCols > 1) {
      val cols = filteredMatrix.numCols
      val sampleSize = filteredMatrix.multiply(new DenseMatrix(cols, 1, Array.fill(cols)(1.0))).values.sum
      val chiSquareRes = Statistics.chiSqTest(filteredMatrix)
      val phiSquare = chiSquareRes.statistic / sampleSize
      val denom = math.min(filteredMatrix.numRows - 1, filteredMatrix.numCols - 1)
      ChiSquaredResults(math.sqrt(phiSquare / denom), chiSquareRes.statistic, chiSquareRes.pValue)
    } else ChiSquaredResults(Double.NaN, Double.NaN, Double.NaN)
  }

  /**
   * Computes the pointwise (and total) mutual information between the given feature values and label values given by
   * the input contingency matrix
   *
   * @param contingency Matrix of co-occurrences of feature values with label values. Each row represents a different
   *                    feature choice, while each column represents a different label value.
   * @return 2-element result tuple containing (Map of label values to pointwise mutual information values,
   *         total mutual information between the feature and the label)
   */
  private[stats] def mutualInfoWithFilter(contingency: Matrix): (LabelWiseValues.Type, Double) = {
    val filteredMatrix = filterEmpties(contingency)
    mutualInfo(filteredMatrix)
  }

  /**
   * Mutual information calculation, assuming the matrix passed in has had all its empty rows and columns filtered out
   *
   * @param contingency Contingency matrix with no rows/cols of zeros
   * @return 2-element result tuple containing (Map of label values to pointwise mutual information values,
   *         total mutual information between the feature and the label)
   */
  private[stats] def mutualInfo(contingency: Matrix): (LabelWiseValues.Type, Double) = {
    // First check contingency matrix to see make sure we can compute a chi-squared test on it
    val (rows, cols) = (contingency.numRows, contingency.numCols)
    // Add all the columns together
    val summedCols = contingency.multiply(new DenseMatrix(cols, 1, Array.fill(cols)(1.0)))
    val sampleSize = summedCols.values.sum
    val summedRows = {
      val fm = contingency match {
        case s: SparseMatrix => s.toDense
        case d: DenseMatrix => d
      }
      new DenseMatrix(1, rows, Array.fill(rows)(1.0)).multiply(fm)
    }

    // This is the column-major array that corresponds to the matrix of PMI values from the contingency matrix
    val pmiArray = {
      for {
        (vec, j) <- contingency.colIter.zipWithIndex
        (v, i) <- vec.toArray.zipWithIndex
      } yield {
        // TODO: Use our SpecialDoubleSerializer to serialize these arrays with Infinity/NaN values
        // TODO: Figure out something better to do with empty rows/cols (eg. make PMI NaN?)
        j -> (if (v == 0 || summedCols(i, 0) == 0 || summedRows(0, j) == 0) 0.0
        else math.log(math.max(v, 1e-99) * sampleSize / (summedCols(i, 0) * summedRows(0, j))) / math.log(2.0))
      }
    }.toArray

    // We really want it to be a map to an array of doubles where the map key is the label (column index) and the
    // value is the array of PMI values for each of the features contained in the contingency matrix
    val pmiMap = pmiArray.groupBy(_._1).map{
      case (label, pmiValues) => label.toString -> pmiValues.map(_._2)
    }

    // The mutual information is also easily calculated from all the pmi values and the contingency matrix
    val mi = pmiArray.map(_._2).zip(contingency.toArray).map { case (pmi, count) => pmi * count / sampleSize }.sum

    pmiMap -> mi
  }

  /**
   * Calculates the max confidence of all the association rules per row. The confidence of the (i,j)th entry of
   * the contingency matrix expresses how often the rule (choice i => label j) is satisfied.
   *
   * @param contingency
   * @return
   */
  private[stats] def maxConfidences(contingency: Matrix): ConfidenceResults = {
    val summedCols = contingency.multiply(new DenseMatrix(contingency.numCols, 1, Array.fill(contingency.numCols)(1.0)))
    // summedCols.values.sum should be equal to the size of the data, so should always be >0 here
    val supports = summedCols.values.map(_ / summedCols.values.sum)
    val maxConfidences = contingency.rowIter.zip(summedCols.values.iterator).map(f => {
      // If there is no support for this choice, just return 0 for the confidence instead of NaN or dividing by zero
      if (f._2 == 0) 0 else f._1(f._1.argmax) / f._2
    }).toArray

    ConfidenceResults(maxConfidences = maxConfidences, supports = supports)
  }

  /**
   * Calculates all of the statistics we use that come from contingency matrices between categorical features
   * and categorical labels and stores them in a ContingencyStats case class.
   *
   * @param contingency Matrix of co-occurrences of feature values with label values. Each row represents a different
   *                    feature choice, while each column represents a different label value.
   * @return ContingencyStats object containing all the statistics we calculate from contingency matrices
   */
  def contingencyStats(contingency: Matrix): ContingencyStats = {
    val filteredMatrix = filterEmpties(contingency)
    // check if filteredMatrix is entirely empty
    if (filteredMatrix.numActives <= 0) {
      ContingencyStats(
        ChiSquaredResults(cramersV = Double.NaN, chiSquaredStat = Double.NaN, pValue = Double.NaN),
        pointwiseMutualInfo = Map.empty,
        contingencyMatrix = Map.empty,
        mutualInfo = Double.NaN,
        confidenceResults = ConfidenceResults(Array.empty, Array.empty)
      )
    }
    else {
      // If we filter out empty columns, then we don't end up with the right number of elements in pmiMap
      val (pmiMap, mi) = mutualInfo(contingency) // mutualInfoOnFiltered(filteredMatrix)
      val cvRes = chiSquaredTestOnFiltered(filteredMatrix)
      val confRes = maxConfidences(contingency)

      ContingencyStats(
        ChiSquaredResults(cramersV = cvRes.cramersV, chiSquaredStat = cvRes.chiSquaredStat, pValue = cvRes.pValue),
        pointwiseMutualInfo = pmiMap,
        contingencyMatrix = matrixToLableWiseStats(contingency),
        mutualInfo = mi,
        confidenceResults = confRes
      )
    }
  }

  /**
   * Same as contingencyStats method, but specialized to MultiPickLists. The standard contingency table stats are
   * not technically valid for MultiPickLists because the choices are not independent from each other (multipicklists
   * are multi-hot encoded instead of one-hot encoded).
   *
   * There are several strategies to deal with this to calculate statistics similar to Cramer's V. We follow
   * https://cran.r-project.org/web/packages/MRCV/vignettes/MRCV-vignette.pdf for inspiration, but use a slightly
   * different scheme where we compute stats from a 2 x numLabels contingency matrix for each choice separately, and
   * take the max of these Cramer's V values (one per choice) as the Cramer's V value for the entire MultiPickList. See
   * BadFeatureZooTest for testing how this performs on different types of relations between MultiPickLists and the
   * label.
   *
   * @param contingency Matrix of co-occurrences of feature values with label values. Each row represents a different
   *                    feature choice, while each column represents a different label value.
   * @param labelCounts Array of counts of each label, used to construct the 2 x numLabels contingency matrices for
   *                    each choice
   * @return ContingencyStats object containing all the statistics we calculate from contingency matrices
   */
  def contingencyStatsFromMultiPickList(contingency: Matrix, labelCounts: Array[Double]): ContingencyStats = {
    // Filter the empty rows and columns out of the matrix
    val filteredMatrix = filterEmpties(contingency)

    val cols = contingency.numCols

    // Iterate through all the rows (multipicklist choices) of the matrix, and construct 2 x numLabels contingency
    // matrices of just that single choice and compute statistics for each of those
    val singleRowStats = filteredMatrix.rowIter.map(row => {
      val singleChoiceValues = row.toArray.zipWithIndex.flatMap { case (x, i) => Array(x, labelCounts(i) - x) }
      val singleChoiceContingencyMatrix = new DenseMatrix(numRows = 2, numCols = cols, values = singleChoiceValues)
      contingencyStats(singleChoiceContingencyMatrix)
    }).toArray

    // Index of contingency matrix stats that has the largest Cramer's V
    val winningIndex = singleRowStats.zipWithIndex.maxBy(_._1.chiSquaredResults.cramersV)._2
    val winningStats = singleRowStats(winningIndex)

    val fullMatrixStats = contingencyStats(contingency)
    /*
    Other potential Cramer's V values we tested, for completeness:
    val sampleSize = filteredMatrix.multiply(new DenseMatrix(cols, 1, Array.fill(cols)(1.0))).values.sum

    val avgCramersV = singleRowStats.map(_.cramersV).sum / singleRowStats.length
    val adjustedCramersV = math.sqrt(avgChiSquaredStat / sampleSize)
    val fullMatrixCramersV = fullMatrixStats.cramersV
     */

    // TODO: Figure out something better to do about PMI and MI for multipicklists than using the full matrix ones
    ContingencyStats(
      chiSquaredResults = winningStats.chiSquaredResults,
      pointwiseMutualInfo = fullMatrixStats.pointwiseMutualInfo,
      contingencyMatrix = matrixToLableWiseStats(contingency),
      mutualInfo = fullMatrixStats.mutualInfo,
      confidenceResults = fullMatrixStats.confidenceResults
    )
  }

}
