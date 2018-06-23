/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of Salesforce.com nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

package com.salesforce.op.stages.base.binary

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Dataset
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class BinaryEstimatorTest
  extends OpEstimatorSpec[OPVector, BinaryModel[Text, Text, OPVector], BinaryEstimator[Text, Text, OPVector]] {

  val (inputData, city, country) = TestFeatureBuilder("city", "country",
    Seq(
      (Text("San Francisco"), Text("USA")),
      (Text("Paris"), Text("France")),
      (Text("Austin"), Text("USA")),
      (Text("San Francisco"), Text("USA")),
      (Text("Paris"), Text("USA")),
      (Text("Puerto Arenas"), Text("Chile")),
      (Text("Iquitos"), Text(None))
    )
  )

  val estimator = new TestPivotEstimator().setInput(city, country)

  val expectedResult = Seq(
    Vectors.dense(1.0, 0.0),
    Vectors.dense(0.0, 1.0),
    Vectors.dense(0.0, 1.0),
    Vectors.dense(1.0, 0.0),
    Vectors.dense(0.0, 1.0),
    Vectors.dense(0.0, 1.0),
    Vectors.dense(0.0, 1.0)
  ).map(_.toOPVector)

}


class TestPivotEstimator(uid: String = UID[TestPivotEstimator])
  extends BinaryEstimator[Text, Text, OPVector](operationName = "pivot", uid = uid) {

  def fitFn(data: Dataset[(Text#Value, Text#Value)]): BinaryModel[Text, Text, OPVector] = {
    import data.sparkSession.implicits._
    val counts =
      data.map { case (cty, cntry) => Seq(cty, cntry).flatten.mkString(" ") -> 1 }
        .groupByKey(_._1).reduceGroups((a, b) => (a._1, a._2 + b._2)).map(_._2)

    val topValue = counts.collect().minBy(-_._2)._1
    new TestPivotModel(topValue = topValue, operationName = operationName, uid = uid)
  }
}
final class TestPivotModel private[op](val topValue: String, operationName: String, uid: String)
  extends BinaryModel[Text, Text, OPVector](operationName = operationName, uid = uid) {

  def transformFn: (Text, Text) => OPVector = (city: Text, country: Text) => {
    val cityCountry = Seq(city.value, country.value).flatten.mkString(" ")
    val vector = if (topValue == cityCountry) Vectors.dense(1, 0) else Vectors.dense(0, 1)
    vector.toOPVector
  }

}
