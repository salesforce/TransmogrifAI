package com.salesforce.op.filters

import com.salesforce.op.features.types.FeatureType

import scala.util.Random

object DistributionTest {

  case class TrivialDistribution(count: Double, nullCount: Double) extends Distribution[TrivialFeatureType] {
    def maximum: Double = 1

    def minimum: Double = 0

    def cdf(x: Double): Double =
      if (x < 0) {
        0.0
      } else if (x <= 0 && x < 1) {
        nullCount / (count + nullCount)
      } else {
        count / (count + nullCount)
      }

    def update(value: TrivialFeatureType): Distribution[TrivialFeatureType] = {
      val outputValue = value.value
      TrivialDistribution(count + outputValue, nullCount + outputValue)
    }
  }

  class TrivialFeatureType extends FeatureType {
    type Value = Double
    def value: Double = if (isEmpty) 1 else 0

    def isEmpty: Boolean = Random.nextInt(2) == 0
  }
}
