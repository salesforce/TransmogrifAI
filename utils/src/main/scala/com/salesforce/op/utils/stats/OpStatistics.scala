/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.stats

import org.apache.spark.mllib.linalg.{DenseMatrix, Matrix, SparseMatrix}
import org.apache.spark.mllib.stat.Statistics

object OpStatistics {

  /**
   * 2-element result tuple containing (Map of label values to pointwise mutual information values,
   * total mutual information between the feature and the label)
   */
  type PointwiseMutualInfo = Map[Int, Array[Double]]

  /**
   * Container class for statistics calculated from contingency tables constructed from categorical variables
   *
   * @param cramersV            Map between feature name in feature vector and Cramer's V value
   * @param pointwiseMutualInfo Map between feature name in feature vector and map of pointwise mutual information
   *                            values between that feature and all values the label can take
   * @param mutualInfo          Map between feature name in feature vector and the mutual information with the label
   */
  case class ContingencyStats(cramersV: Double, pointwiseMutualInfo: PointwiseMutualInfo, mutualInfo: Double)

  /**
   * Filters out empty rows and columns from the contingency matrix
   *
   * @param contingency
   * @return
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
   * Compute Cramer's V statistic based on the given contingency matrix and sample size. This will filter out the
   * empty rows/cols from the contingency matrix before calculating Cramer's V since the chi-squared statistic
   * used inside the Cramer's V calculation cannot be done on matrices with any rows/cols of all zeros
   *
   * @see https://goo.gl/Fw6ZMN
   *
   * @param contingency Contingency matrix
   * @return Cramer's V value for that contingency matrix
   */
  private[stats] def cramersV(contingency: Matrix): Double = {
    val filteredMatrix = filterEmpties(contingency)
    cramersVOnFiltered(filteredMatrix)
  }

  /**
   * Cramer's V calculation, assuming the matrix passed in has had all its empty rows and columns filtered out. It
   * will return Double.NaN if the contingency matrix does not have 2 or more rows/cols (each).
   *
   * @param filteredMatrix Contingency matrix with no rows/cols of zeros
   * @return Cramer's V value for that contingency matrix
   */
  private[stats] def cramersVOnFiltered(filteredMatrix: Matrix): Double = {
    if (filteredMatrix.numRows > 1 && filteredMatrix.numCols > 1) {
      val cols = filteredMatrix.numCols
      val sampleSize = filteredMatrix.multiply(new DenseMatrix(cols, 1, Array.fill(cols)(1.0))).values.sum
      val chiSquare = Statistics.chiSqTest(filteredMatrix).statistic
      val phiSquare = chiSquare / sampleSize
      val denom = math.min(filteredMatrix.numRows - 1, filteredMatrix.numCols - 1)
      math.sqrt(phiSquare / denom)
    } else Double.NaN
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
  private[stats] def mutualInfoWithFilter(contingency: Matrix): (PointwiseMutualInfo, Double) = {
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
  private[stats] def mutualInfo(contingency: Matrix): (PointwiseMutualInfo, Double) = {
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
    val pmiMap = pmiArray.groupBy(_._1).mapValues(_.map(_._2))

    // The mutual information is also easily calculated from all the pmi values and the contingency matrix
    val mi = pmiArray.map(_._2).zip(contingency.toArray).map { case (pmi, count) => pmi * count / sampleSize }.sum

    pmiMap -> mi
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
      return ContingencyStats(cramersV = Double.NaN, pointwiseMutualInfo = Map.empty, mutualInfo = 0.0)
    }
    // If we filter out empty columns, then we don't end up with the right number of elements in pmiMap
    val (pmiMap, mi) = mutualInfo(contingency) // mutualInfoOnFiltered(filteredMatrix)
    val cvMap = cramersVOnFiltered(filteredMatrix)

    ContingencyStats(cramersV = cvMap, pointwiseMutualInfo = pmiMap, mutualInfo = mi)
  }

  /**
   * Same as above method, except in this case, we are only passed one column, but have a 2x2 matrix. We need to
   * return just one column of PMI values in order for the feature names to match the values in the PMI arrays.
   * This function will only return the PMI values for the nullIndicator column
   *
   * @param contingency Matrix of co-occurrences of feature values with label values. Each row represents a different
   *                    feature choice, while each column represents a different label value.
   * @return ContingencyStats object containing all the statistics we calculate from contingency matrices
   */
  def contingencyStatsFromSingleColumn(contingency: Matrix): ContingencyStats = {
    val res = contingencyStats(contingency)
    // The first row of the contingency matrix (first element of each column) is the actual content of the
    // nullIndicator column so only use those PMI values
    val truncatedPMI = res.pointwiseMutualInfo.mapValues(v => Array(v.head))

    ContingencyStats(cramersV = res.cramersV, pointwiseMutualInfo = truncatedPMI, mutualInfo = res.mutualInfo)
  }

}
