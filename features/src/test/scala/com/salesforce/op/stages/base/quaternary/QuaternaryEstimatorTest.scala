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

package com.salesforce.op.stages.base.quaternary

import com.salesforce.op.UID
import com.salesforce.op.features.types._
import com.salesforce.op.test.{OpEstimatorSpec, TestFeatureBuilder}
import org.apache.spark.sql.Dataset
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class QuaternaryEstimatorTest
  extends OpEstimatorSpec[Real,
    QuaternaryModel[Real, TextMap, BinaryMap, MultiPickList, Real],
    QuaternaryEstimator[Real, TextMap, BinaryMap, MultiPickList, Real]] {

  val (inputData, reals, textMap, booleanMap, binary) = TestFeatureBuilder(
    Seq(
      (Real.empty, TextMap(Map("a" -> "keen")), BinaryMap(Map("a" -> true)), MultiPickList(Set("a"))),
      (Real(15.0), TextMap(Map("b" -> "bok")), BinaryMap(Map("b" -> true)), MultiPickList(Set("b"))),
      (Real(23.0), TextMap(Map("c" -> "bar")), BinaryMap(Map("c" -> true)), MultiPickList(Set("c"))),
      (Real(40.0), TextMap(Map.empty), BinaryMap(Map("d" -> true)), MultiPickList(Set("d"))),
      (Real(65.0), TextMap(Map("e" -> "B")), BinaryMap(Map("e" -> true)), MultiPickList(Set("e")))
    )
  )

  val estimator = new FantasticFourEstimator().setInput(reals, textMap, booleanMap, binary)

  val expectedResult = Seq(Real.empty, Real(-31.6), Real(-23.6), Real.empty, Real(18.4))
}

class FantasticFourEstimator(uid: String = UID[FantasticFourEstimator])
  extends QuaternaryEstimator[Real, TextMap, BinaryMap, MultiPickList, Real](operationName = "fantasticFour", uid = uid)
    with FantasticFour  {

  // scalastyle:off line.size.limit
  def fitFn(dataset: Dataset[(Real#Value, TextMap#Value, BinaryMap#Value, MultiPickList#Value)]): QuaternaryModel[Real, TextMap, BinaryMap, MultiPickList, Real] = {
    import dataset.sparkSession.implicits._
    val topAge = dataset.map(_._1.getOrElse(0.0)).groupBy().max().first().getDouble(0)
    val mean = dataset.map { case (age, strMp, binMp, gndr) =>
      if (filterFN(age, strMp, binMp, gndr)) age.getOrElse(topAge) else topAge
    }.groupBy().mean().first().getDouble(0)

    new FantasticFourModel(mean = mean, operationName = operationName, uid = uid)
  }
  // scalastyle:on

}

final class FantasticFourModel private[op](val mean: Double, operationName: String, uid: String)
  extends QuaternaryModel[Real, TextMap, BinaryMap, MultiPickList, Real](operationName = operationName, uid = uid)
    with FantasticFour {

  def transformFn: (Real, TextMap, BinaryMap, MultiPickList) => Real = (age, strMp, binMp, gndr) => new Real(
    if (filterFN(age.v, strMp.v, binMp.v, gndr.v)) Some(age.v.get - mean) else None
  )

}

sealed trait FantasticFour {
  def filterFN(a: Real#Value, sm: TextMap#Value, bm: BinaryMap#Value, g: MultiPickList#Value): Boolean =
    a.nonEmpty && g.nonEmpty && sm.contains(g.head) && bm.contains(g.head)
}

