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

package com.salesforce.op.stages.base.ternary

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import org.apache.spark.sql.Dataset
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class TernaryEstimatorTest
  extends OpEstimatorSpec[Real,
    TernaryModel[MultiPickList, Binary, RealMap, Real],
    TernaryEstimator[MultiPickList, Binary, RealMap, Real]] {

  val (inputData, gender, numericMap, survived) = TestFeatureBuilder("gender", "numericMap", "survived",
    Seq(
      (MultiPickList.empty, RealMap(Map("teen" -> 1.0)), Binary(true)),
      (MultiPickList(Set("teen")), RealMap(Map("teen" -> 2.0)), Binary(false)),
      (MultiPickList(Set("teen")), RealMap(Map("teen" -> 3.0)), Binary(false)),
      (MultiPickList(Set("adult")), RealMap(Map("adult" -> 1.0)), Binary(false)),
      (MultiPickList(Set("senior")), RealMap(Map("senior" -> 1.0, "adult" -> 2.0)), Binary(false))
    )
  )

  val estimator = new TripleInteractionsEstimator().setInput(gender, survived, numericMap)

  val expectedResult = Seq(Real.empty, Real(0.25), Real(1.25), Real(-0.75), Real(-0.75))
}

class TripleInteractionsEstimator(uid: String = UID[TripleInteractionsEstimator])
  extends TernaryEstimator[MultiPickList, Binary, RealMap, Real](operationName = "tripleInteractions", uid = uid)
    with TripleInteractions {

  // scalastyle:off line.size.limit
  def fitFn(dataset: Dataset[(MultiPickList#Value, Binary#Value, RealMap#Value)]): TernaryModel[MultiPickList, Binary, RealMap, Real] = {
    import dataset.sparkSession.implicits._
    val mean = {
      dataset.map { case (gndr, srvvd, nmrcMp) =>
        if (survivedAndMatches(gndr, srvvd, nmrcMp)) nmrcMp(gndr.head) else 0.0
      }.filter(_ != 0.0).groupBy().mean().first().getDouble(0)
    }
    new TripleInteractionsModel(mean = mean, operationName = operationName, uid = uid)
  }
  // scalastyle:on

}

final class TripleInteractionsModel private[op](val mean: Double, operationName: String, uid: String)
  extends TernaryModel[MultiPickList, Binary, RealMap, Real](operationName = operationName, uid = uid)
    with TripleInteractions {

  def transformFn: (MultiPickList, Binary, RealMap) => Real = (g: MultiPickList, s: Binary, nm: RealMap) => new Real(
    if (!survivedAndMatches(g.value, s.value, nm.value)) None
    else Some(nm.value(g.value.head) - mean)
  )

}

sealed trait TripleInteractions {
  def survivedAndMatches(g: MultiPickList#Value, s: Binary#Value, nm: RealMap#Value): Boolean =
    !s.getOrElse(false) && g.nonEmpty && nm.contains(g.head)
}
