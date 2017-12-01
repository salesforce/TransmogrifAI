/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.utils.stats

import com.salesforce.op.test.TestCommon
import com.salesforce.op.utils.stats.OpStatistics.ContingencyStats
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
    val res = OpStatistics.cramersV(contingencyMatrix)

    math.abs(res - 0.2921) < tol shouldBe true
  }

  it should "correctly calculate Cramer's V for a 4x4 matrix" in {
    val contingencyMatrix = new DenseMatrix(4, 4, Array[Double](5, 15, 20, 68, 29, 54, 84, 119,
      14, 14, 17, 26, 16, 10, 94, 7))
    val res = OpStatistics.cramersV(contingencyMatrix)

    math.abs(res - 0.279) < tol shouldBe true
  }

  it should "correctly calculate Cramer's V for a 3x6 matrix" in {
    val contingencyMatrix = new DenseMatrix(3, 6, Array[Double](192, 221, 229, 185, 202, 194, 62, 199, 97,
      78, 30, 44, 78, 29, 53, 21, 18, 18))
    val res = OpStatistics.cramersV(contingencyMatrix)

    math.abs(res - 0.1815) < tol shouldBe true
  }

  it should "correctly filter out empty columns from the Cramer's V calculation and not throw an exception" in {
    val contingencyMatrix = new DenseMatrix(5, 2, Array[Double](0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 1.0, 2.0, 7.0))
    val res = OpStatistics.cramersV(contingencyMatrix)

    res.isNaN shouldBe true
  }

  it should "correctly filter out empty rows from the Cramer's V calculation and not throw an exception" in {
    val contingencyMatrix = new DenseMatrix(6, 2, Array[Double](0, 757, 0, 0, 726, 0, 0, 731, 0, 0, 2621, 0))
    val res = OpStatistics.cramersV(contingencyMatrix)

    math.abs(res - 0.2921) < tol shouldBe true
  }

  it should "correctly filter out empty rows & cols from the Cramer's V calculation and not throw an exception" in {
    val contingencyMatrix = new DenseMatrix(4, 4, Array[Double](0, 0, 0, 0, 0, 757, 726, 0,
      0, 0, 0, 0, 0, 731, 2621, 0))
    val res = OpStatistics.cramersV(contingencyMatrix)

    math.abs(res - 0.2921) < tol shouldBe true
  }

  it should "not fail when passed an empty matrix" in {
    val contingencyMatrix = new DenseMatrix(0, 0, Array.empty[Double])
    val res = OpStatistics.cramersV(contingencyMatrix)

    res.isNaN shouldBe true
  }

  it should "correctly calculate pmi on a 2x2 matrix" in {
    val contingencyMatrix = new DenseMatrix(2, 2, Array[Double](10, 15, 70, 5))
    val resMap = Map(0 -> Array(-1.0, 1.5850), 1 -> Array(0.2224, -1.5850))
    val (pmiMap, mi) = OpStatistics.mutualInfo(contingencyMatrix)

    resMap.keySet shouldBe pmiMap.keySet
    pmiMap.keys.forall(k => pmiMap(k).zip(resMap(k)).forall(v => math.abs(v._1 - v._2) < tol))
    math.abs(mi - 0.2142) < tol shouldBe true
  }

  it should "correctly filter out empty columns from the PMI calculation and not throw an exception" in {
    val contingencyMatrix = new DenseMatrix(5, 2, Array[Double](0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 1.0, 2.0, 7.0))
    val (pmiMap, mi) = OpStatistics.mutualInfo(contingencyMatrix)

    // PMI should correspond to just a single label, so all the values should be zero
    pmiMap.values.forall(_.forall(_ == 0)) shouldBe true
    mi shouldBe 0
  }

  it should "correctly filter out empty rows from the PMI calculation and not throw an exception" in {
    val contingencyMatrix = new DenseMatrix(6, 2, Array[Double](0, 10, 0, 0, 15, 0, 0, 70, 0, 0, 5, 0))
    val resMap = Map(0 -> Array(-1.0, 1.5850), 1 -> Array(0.2224, -1.5850))
    val (pmiMap, mi) = OpStatistics.mutualInfo(contingencyMatrix)

    resMap.keySet shouldBe pmiMap.keySet
    pmiMap.keys.forall(k => pmiMap(k).zip(resMap(k)).forall(v => math.abs(v._1 - v._2) < tol))
    math.abs(mi - 0.2142) < tol shouldBe true
  }

  it should "correctly filter out empty rows & cols from the PMI calculation and not throw an exception" in {
    val contingencyMatrix = new DenseMatrix(4, 4, Array[Double](0, 0, 0, 0, 0, 10, 15, 0,
      0, 0, 0, 0, 0, 70, 5, 0))
    val resMap = Map(0 -> Array(-1.0, 1.5850), 1 -> Array(0.2224, -1.5850))
    val (pmiMap, mi) = OpStatistics.mutualInfo(contingencyMatrix)

    resMap.keySet shouldBe pmiMap.keySet
    pmiMap.keys.forall(k => pmiMap(k).zip(resMap(k)).forall(v => math.abs(v._1 - v._2) < tol))
    math.abs(mi - 0.2142) < tol shouldBe true
  }

  it should "return contingencyStats objects from contingencyStats, even when empty" in {
    val contingencyMatrix = new DenseMatrix(0, 0, Array.empty[Double])
    val res = OpStatistics.contingencyStats(contingencyMatrix)

    res.cramersV.isNaN shouldBe true
    res.pointwiseMutualInfo shouldBe Map.empty[Int, Array[Double]]
    res.mutualInfo shouldBe 0.0
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
      val res = OpStatistics.cramersV(matrix)
      if (matrix.numRows > 1 && matrix.numCols > 1) {
        res >= 0 shouldBe true
        res <= 1 shouldBe true
      } else {
        res.isNaN shouldBe true
      }
    }
  }

}
