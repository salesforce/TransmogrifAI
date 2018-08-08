package com.salesforce.op.filters.distrib

import com.salesforce.op.features.types.Binary
import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalactic.TolerantNumerics
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

import scala.util.Random

@RunWith(classOf[JUnitRunner])
class DistributionTest extends FlatSpec with TestCommon {

  implicit val doubleEquality = TolerantNumerics.tolerantDoubleEquality(0.00001)

  Spec(BinomialDistribution.getClass) should "work" in {
    val testDist =
      Seq(Option(true), Option(false), None, None, Option(true)).map(Binary(_))
        .foldLeft(BinomialDistribution(0, 0, 0)) { (dist, binary) => dist.update(binary) }

        testDist.cdf(0) shouldEqual 1.0 / 3
        testDist.cdf(1) shouldEqual 1.0

        testDist.mass(0) shouldEqual 1.0 / 3
        testDist.mass(1) shouldEqual 2.0 / 3
  }

  case class BinomialDistribution(count: Double, nullCount: Double, successCount: Double)
    extends DiscreteDistribution[Binary] {

    private val p: Double = if (count == 0) 0 else successCount / count

    def maximum: Double = count

    def minimum: Double = 0

    def cdf(x: Double): Double = x match {
      case d if d < 0 => 0
      case d if d < 1 => 1 - p
      case _ => 1
    }

    def mass(x: Double) = x match {
      case 1 => p
      case 0 => 1 - p
      case _ => 0
    }

    def values: Set[Double] = Set(0, 1)

    def update(value: Binary): BinomialDistribution =
      value.value match {
        case Some(true) => BinomialDistribution(count + 1, nullCount, successCount + 1)
        case Some(false) => BinomialDistribution(count + 1, nullCount, successCount)
        case None => BinomialDistribution(count, nullCount + 1, successCount)
      }
  }
}
