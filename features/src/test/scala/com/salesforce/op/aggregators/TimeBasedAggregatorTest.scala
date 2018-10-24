package com.salesforce.op.aggregators

import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class TimeBasedAggregatorTest extends FlatSpec with TestCommon {

  val data = Seq()

  Spec[MostRecentAggregator[_]] should "return the most recent event" in {
    val featureAggregator =
      GenericFeatureAggregator(
        extractFn = extractFn,
        aggregator = aggregator,
        isResponse = false,
        specialTimeWindow = None
      )
  }

  it should "return the most recent event within the time window" in {

  }

  it should "return the feature type empty value when no events are passed in" in {

  }

  Spec[FirstAggregator[_]] should "return the first event" in {

  }

  it should "return the first event within the time window" in {

  }

  it should "return the feature type empty value when no events are passed in" in {

  }
}
