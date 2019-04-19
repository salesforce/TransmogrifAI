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

package com.salesforce.op.test

import com.salesforce.op.features.{FeatureLike, FeatureSparkTypes}
import com.salesforce.op.features.types._
import com.salesforce.op.utils.spark.RichRow._
import org.apache.spark.sql.DataFrame
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class TestFeatureBuilderTest extends FlatSpec with TestSparkContext with FeatureAsserts {

  Spec(TestFeatureBuilder.getClass)  should "create a dataset with one feature" in {
    val res@(ds, f1) = TestFeatureBuilder[Real](Seq(Real(1), Real(2L), Real(3.1f), Real(4.5)))

    assertFeature(f1)(name = "f1", in = ds.head(), out = Real(1.0))
    assertResults(ds, res, expected = Seq(1, 2L, 3.1f, 4.5))
  }

  it should "create a dataset with two features" in {
    val res@(ds, f1, f2) = TestFeatureBuilder(
      Seq[(Text, Integral)](
        (Text("one"), Integral(1)), (Text("two"), Integral(2)), (Text("NULL"), Integral.empty)
      )
    )
    assertFeature(f1)(name = "f1", in = ds.head(), out = Text("one"))
    assertFeature(f2)(name = "f2", in = ds.head(), out = Integral(1))
    assertResults(ds, res, expected = Seq(("one", 1), ("two", 2), ("NULL", null)))
  }

  case class TFBClassTest(s: Text, x: Integral)

  it should "create a dataset with two features with a case class" in {
    val res@(ds, f1, f2) = TestFeatureBuilder[Text, Integral](
      Seq(
        TFBClassTest(Text("one"), Integral(1)),
        TFBClassTest(Text("two"), Integral(2)),
        TFBClassTest(Text("NULL"), Integral.empty)
      ).flatMap(TFBClassTest.unapply)
    )
    assertResults(ds, res, expected = Seq(("one", 1), ("two", 2), ("NULL", null)))
  }

  it should "create a dataset with three features" in {
    val res@(ds, f1, f2, f3) = TestFeatureBuilder(Seq[(Text, Integral, Real)](
      (Text("one"), Integral(1), Real(1.0)),
      (Text("two"), Integral(2), Real(2.3)),
      (Text("NULL"), Integral.empty, Real.empty)
    ))
    f1.name shouldBe "f1"
    f1.typeName shouldBe FeatureType.typeName[Text]
    f2.name shouldBe "f2"
    f2.typeName shouldBe FeatureType.typeName[Integral]
    f3.name shouldBe "f3"
    f3.typeName shouldBe FeatureType.typeName[Real]

    assertFeature(f1)(name = "f1", in = ds.head(), out = Text("one"))
    assertFeature(f2)(name = "f2", in = ds.head(), out = Integral(1))
    assertFeature(f3)(name = "f3", in = ds.head(), out = Real(1.0))
    assertResults(ds, res, expected = Seq(("one", 1, 1.0), ("two", 2, 2.3), ("NULL", null, null)))
  }

  it should "create a dataset with four features" in {
    val res@(ds, f1, f2, f3, f4) = TestFeatureBuilder(
      Seq[(Text, Integral, Real, Integral)](
        (Text("one"), Integral(1), Real(1.0), Integral(-1)),
        (Text("two"), Integral(2), Real(2.3), Integral(1)),
        (Text("NULL"), Integral.empty, Real.empty, Integral(1))
      )
    )
    assertFeature(f1)(name = "f1", in = ds.head(), out = Text("one"))
    assertFeature(f2)(name = "f2", in = ds.head(), out = Integral(1))
    assertFeature(f3)(name = "f3", in = ds.head(), out = Real(1.0))
    assertFeature(f4)(name = "f4", in = ds.head(), out = Integral(-1))
    assertResults(ds, res, expected = Seq(("one", 1, 1.0, -1), ("two", 2, 2.3, 1), ("NULL", null, null, 1)))
  }

  it should "create a dataset with five features" in {
    val res@(ds, f1, f2, f3, f4, f5) = TestFeatureBuilder(
      Seq[(Text, Integral, Real, Integral, MultiPickList)](
        (Text("one"), Integral(1), Real(1.0), Integral(-1), MultiPickList(Set("1", "2", "2"))),
        (Text("two"), Integral(2), Real(2.3), Integral(1), MultiPickList(Set("3", "4")))
      )
    )
    assertFeature(f1)(name = "f1", in = ds.head(), out = Text("one"))
    assertFeature(f2)(name = "f2", in = ds.head(), out = Integral(1))
    assertFeature(f3)(name = "f3", in = ds.head(), out = Real(1.0))
    assertFeature(f4)(name = "f4", in = ds.head(), out = Integral(-1))
    assertFeature(f5)(name = "f5", in = ds.head(), out = MultiPickList(Set("1", "2", "2")))
    assertResults(ds, res, expected = Seq(("one", 1, 1.0, -1, List("1", "2")), ("two", 2, 2.3, 1, List("3", "4"))))
  }

  it should "create a dataset with arbitrary amount of features" in {
    val (ds, features) = TestFeatureBuilder(
      Seq(Real(0.0)), Seq(Text("a")), Seq(Integral(5L)), Seq(Real(1.0)), Seq(Text("b")),
      Seq(MultiPickList(Set("3", "4"))), Seq(Real(-3.0))
    )
    features.length shouldBe 7
    ds.count() shouldBe 1
    ds.schema.fields.map(f => f.name -> f.dataType) should contain theSameElementsInOrderAs
      features.map(f => f.name -> FeatureSparkTypes.sparkTypeOf(f.wtt))

    assertFeature(features(0).asInstanceOf[FeatureLike[Real]])(name = "f1", in = ds.head(), out = Real(0.0))
    assertFeature(features(1).asInstanceOf[FeatureLike[Text]])(name = "f2", in = ds.head(), out = Text("a"))
    assertFeature(features(2).asInstanceOf[FeatureLike[Integral]])(name = "f3", in = ds.head(), out = Integral(5L))
    assertFeature(features(3).asInstanceOf[FeatureLike[Real]])(name = "f4", in = ds.head(), out = Real(1.0))
    assertFeature(features(4).asInstanceOf[FeatureLike[Text]])(name = "f5", in = ds.head(), out = Text("b"))
    assertFeature(features(5).asInstanceOf[FeatureLike[MultiPickList]])(
      name = "f6", in = ds.head(), out = MultiPickList(Set("3", "4")))
    assertFeature(features(6).asInstanceOf[FeatureLike[Real]])(name = "f7", in = ds.head(), out = Real(-3.0))
  }

  it should "create a dataset with all random features" in {
    val numOfRows = 15
    val (ds, features) = TestFeatureBuilder.random(numOfRows = numOfRows)()
    features.length shouldBe 51

    ds.schema.fields.map(f => f.name -> f.dataType) should contain theSameElementsInOrderAs
      features.map(f => f.name -> FeatureSparkTypes.sparkTypeOf(f.wtt))

    ds.count() shouldBe numOfRows
  }

  it should "error creating a dataset with invalid number of rows" in {
    the[IllegalArgumentException] thrownBy TestFeatureBuilder.random(numOfRows = 0)()
    the[IllegalArgumentException] thrownBy TestFeatureBuilder.random(numOfRows = -1)()
    the[IllegalArgumentException] thrownBy TestFeatureBuilder(Seq.empty[Real],
      Seq.empty[Real], Seq.empty[Real], Seq.empty[Real], Seq.empty[Real], Seq.empty[Real])
  }

  private def assertResults(ds: DataFrame, res: Product, expected: Traversable[Any]): Unit = {
    val features = res.productIterator.collect { case f: FeatureLike[_] => f }.toArray

    ds.schema.fields.map(f => f.name -> f.dataType) should contain theSameElementsInOrderAs
      features.map(f => f.name -> FeatureSparkTypes.sparkTypeOf(f.wtt))

    ds.collect().map(row => features.map(f => row.getAny(f.name))) should contain theSameElementsInOrderAs
      expected.map { case v: Product => v; case v => Tuple1(v) }.map(_.productIterator.toArray)
  }

}
