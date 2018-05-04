/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.stats

import com.salesforce.op.test.TestCommon
import org.apache.spark.mllib.linalg.DenseMatrix
import org.junit.runner.RunWith
import org.scalacheck.Gen
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks
import org.scalatest.{FlatSpec, PropSpec}

@RunWith(classOf[JUnitRunner])
class OpStatisticsTest extends FlatSpec with TestCommon {
  val tol: Double = 0.001

  Spec(OpStatistics.getClass) should "correctly calculate Cramer's V for a 2x2 matrix" in {
    val contingencyMatrix = new DenseMatrix(2, 2, Array[Double](757.0, 726.0, 731.0, 2621.0))
    val res = OpStatistics.chiSquaredTest(contingencyMatrix).cramersV

    math.abs(res - 0.2921) < tol shouldBe true
  }

  it should "Lower Cramer's V when we add a duplicate row of noise to a matrix that is upper diagonal" in {
    val contingencyMatrix = new DenseMatrix(3, 2, Array[Double](100, 0, 50, 0, 100, 50))
    val res = OpStatistics.chiSquaredTest(contingencyMatrix).cramersV
    val contingencyMatrix2 = new DenseMatrix(4, 2, Array[Double](100, 0, 50, 50, 0, 100, 50, 50))
    val res2 = OpStatistics.chiSquaredTest(contingencyMatrix2).cramersV
    res > res2 shouldBe true
  }

  it should "correctly calculate Cramer's V for a 4x4 matrix" in {
    val contingencyMatrix = new DenseMatrix(4, 4, Array[Double](5, 15, 20, 68, 29, 54, 84, 119,
      14, 14, 17, 26, 16, 10, 94, 7))
    val res = OpStatistics.chiSquaredTest(contingencyMatrix).cramersV

    math.abs(res - 0.279) < tol shouldBe true
  }

  it should "correctly calculate Cramer's V for a 3x6 matrix" in {
    val contingencyMatrix = new DenseMatrix(3, 6, Array[Double](192, 221, 229, 185, 202, 194, 62, 199, 97,
      78, 30, 44, 78, 29, 53, 21, 18, 18))
    val res = OpStatistics.chiSquaredTest(contingencyMatrix).cramersV

    math.abs(res - 0.1815) < tol shouldBe true
  }

  it should "correctly filter out empty columns from the Cramer's V calculation and not throw an exception" in {
    val contingencyMatrix = new DenseMatrix(5, 2, Array[Double](0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 1.0, 2.0, 7.0))
    val res = OpStatistics.chiSquaredTest(contingencyMatrix).cramersV

    res.isNaN shouldBe true
  }

  it should "correctly filter out empty rows from the Cramer's V calculation and not throw an exception" in {
    val contingencyMatrix = new DenseMatrix(6, 2, Array[Double](0, 757, 0, 0, 726, 0, 0, 731, 0, 0, 2621, 0))
    val res = OpStatistics.chiSquaredTest(contingencyMatrix).cramersV

    math.abs(res - 0.2921) < tol shouldBe true
  }

  it should "correctly filter out empty rows & cols from the Cramer's V calculation and not throw an exception" in {
    val contingencyMatrix = new DenseMatrix(4, 4, Array[Double](0, 0, 0, 0, 0, 757, 726, 0,
      0, 0, 0, 0, 0, 731, 2621, 0))
    val res = OpStatistics.chiSquaredTest(contingencyMatrix).cramersV

    math.abs(res - 0.2921) < tol shouldBe true
  }

  it should "not fail when passed an empty matrix" in {
    val contingencyMatrix = new DenseMatrix(0, 0, Array.empty[Double])
    val res = OpStatistics.chiSquaredTest(contingencyMatrix).cramersV

    res.isNaN shouldBe true
  }

  it should "correctly calculate pmi on a 2x2 matrix" in {
    val contingencyMatrix = new DenseMatrix(2, 2, Array[Double](10, 15, 70, 5))
    val resMap = Map(0 -> Array(-1.0, 1.5850), 1 -> Array(0.2224, -1.5850))
    val (pmiMap, mi) = OpStatistics.mutualInfoWithFilter(contingencyMatrix)

    resMap.keySet shouldBe pmiMap.keySet.map(_.toInt)
    pmiMap.keys.forall(k => pmiMap(k).zip(resMap(k.toInt)).forall(v => math.abs(v._1 - v._2) < tol))
    math.abs(mi - 0.2142) < tol shouldBe true
  }

  it should "correctly filter out empty columns from the PMI calculation and not throw an exception" in {
    val contingencyMatrix = new DenseMatrix(5, 2, Array[Double](0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 1.0, 2.0, 7.0))
    val (pmiMap, mi) = OpStatistics.mutualInfoWithFilter(contingencyMatrix)

    // PMI should correspond to just a single label, so all the values should be zero
    pmiMap.values.forall(_.forall(_ == 0)) shouldBe true
    mi shouldBe 0
  }

  it should "correctly filter out empty rows from the PMI calculation and not throw an exception" in {
    val contingencyMatrix = new DenseMatrix(6, 2, Array[Double](0, 10, 0, 0, 15, 0, 0, 70, 0, 0, 5, 0))
    val resMap = Map(0 -> Array(-1.0, 1.5850), 1 -> Array(0.2224, -1.5850))
    val (pmiMap, mi) = OpStatistics.mutualInfoWithFilter(contingencyMatrix)

    resMap.keySet shouldBe pmiMap.keySet.map(_.toInt)
    pmiMap.keys.forall(k => pmiMap(k).zip(resMap(k.toInt)).forall(v => math.abs(v._1 - v._2) < tol))
    math.abs(mi - 0.2142) < tol shouldBe true
  }

  it should "correctly filter out empty rows & cols from the PMI calculation and not throw an exception" in {
    val contingencyMatrix = new DenseMatrix(4, 4, Array[Double](0, 0, 0, 0, 0, 10, 15, 0,
      0, 0, 0, 0, 0, 70, 5, 0))
    val resMap = Map(0 -> Array(-1.0, 1.5850), 1 -> Array(0.2224, -1.5850))
    val (pmiMap, mi) = OpStatistics.mutualInfoWithFilter(contingencyMatrix)

    resMap.keySet shouldBe pmiMap.keySet.map(_.toInt)
    pmiMap.keys.forall(k => pmiMap(k).zip(resMap(k.toInt)).forall(v => math.abs(v._1 - v._2) < tol))
    math.abs(mi - 0.2142) < tol shouldBe true
  }

  it should "return contingencyStats objects from contingencyStats, even when empty" in {
    val contingencyMatrix = new DenseMatrix(0, 0, Array.empty[Double])
    val res = OpStatistics.contingencyStats(contingencyMatrix)

    res.chiSquaredResults.cramersV.isNaN shouldBe true
    res.pointwiseMutualInfo shouldBe Map.empty[Int, Array[Double]]
    res.mutualInfo.isNaN shouldBe true
  }

  it should "correctly calculate max confidences and supports for all the rows on a 2x2 matrix" in {
    val contingencyMatrix = new DenseMatrix(2, 2, Array[Double](0.0, 500.0, 121.0, 688.0))
    val res = OpStatistics.maxConfidences(contingencyMatrix)

    res.maxConfidences.length shouldBe contingencyMatrix.numRows
    math.abs(res.maxConfidences(0) - 1.0) < tol shouldBe true
    math.abs(res.maxConfidences(1) - 0.5791) < tol shouldBe true

    res.supports.length shouldBe contingencyMatrix.numRows
    math.abs(res.supports(0) - 0.0924) < tol shouldBe true
    math.abs(res.supports(1) - 0.9076) < tol shouldBe true
  }

  it should "correctly calculate max confidences and supports for all the rows on a 6x4 matrix" in {
    val contingencyMatrix = new DenseMatrix(6, 4, Array[Double](
      132.0, 0.0, 189.0, 688.0, 321.0, 0.0,
      98.0, 0.0, 823.0, 223.0, 0.0, 366.0,
      123.0, 14.0, 0.0, 119.0, 0.0, 482.0,
      0.0, 0.0, 443.0, 18.0, 321.0, 0.0)
    )
    val res = OpStatistics.maxConfidences(contingencyMatrix)
    val expectedMaxConfidences = Array(0.3739, 1.0, 0.5656, 0.6564, 0.5, 0.5684)
    val expectedSupports = Array(0.0810, 0.0032, 0.3337, 0.2404, 0.1472, 0.1945)

    res.maxConfidences.length shouldBe contingencyMatrix.numRows
    res.maxConfidences.zip(expectedMaxConfidences).foreach(f => {
      math.abs(f._1 - f._2) < tol shouldBe true
    })

    res.supports.length shouldBe contingencyMatrix.numRows
    res.supports.zip(expectedSupports).foreach(f => {
      math.abs(f._1 - f._2) < tol shouldBe true
    })
  }
}

@RunWith(classOf[JUnitRunner])
class OpStatisticsPropertyTest extends PropSpec with TestCommon with PropertyChecks {

  val genInt = Gen.posNum[Int]
  private def genArray(n: Int) = Gen.containerOfN[Array, Int](n, genInt)

  val genMatrix = for {
    rowSize <- Gen.choose(1, 13)
    colSize <- Gen.choose(1, 13)
    size = rowSize * colSize
    array <- genArray(size)
  } yield {
    new DenseMatrix(rowSize, colSize, array.map(_.toDouble))
  }

  property("cramerV function should produce results in expected ranges") {
    forAll(genMatrix) { (matrix: DenseMatrix) =>
      val res = OpStatistics.chiSquaredTest(matrix).cramersV
      if (matrix.numRows > 1 && matrix.numCols > 1) {
        res >= 0 shouldBe true
        res <= 1 shouldBe true
      } else {
        res.isNaN shouldBe true
      }
    }
  }

}
