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

package com.salesforce.op.features

import com.salesforce.op._
import com.salesforce.op.features.types._
import com.salesforce.op.test.{PassengerFeaturesTest, _}
import org.junit.runner.RunWith
import org.scalatest.WordSpec
import org.scalatest.junit.JUnitRunner

import scala.util.{Failure, Success}


// scalastyle:off indentation
@RunWith(classOf[JUnitRunner])
class FeaturesTest extends WordSpec with PassengerFeaturesTest with TestCommon {

  Spec[Feature[_]] when {
    // will not test all edge cases of equals method. That would be too many combinations
    "equals" should {
      "returns true when comparing a feature after copying" in {
        val f1 = height + weight
        val f2 = f1.asInstanceOf[Feature[Real]].copy()
        f1.equals(f2) shouldBe true
        f2.equals(f1) shouldBe true
      }
      "not be commutative in parent ordering" in {
        val f1 = height + weight
        val f2 = f1.asInstanceOf[Feature[Real]].copy(
          parents = f1.parents.tail :+ f1.parents.head
        )
        f1.equals(f2) shouldBe false
        f2.equals(f1) shouldBe false
      }
      "be false when comparing different features" in {
        age.equals(weight) shouldBe false
        age == weight shouldBe false
        weight.equals(age) shouldBe false
        gender.equals(age) shouldBe false
      }
    }
    "hashcode" should {
      "be overridden to return the uid" in {
        age.hashCode shouldBe age.uid.hashCode
      }
    }
    "name" should {
      "be the same as val name" in {
        age.name shouldBe "age"
        gender.name shouldBe "gender"
        height.name shouldBe "height"
        weight.name shouldBe "weight"
        description.name shouldBe "description"
        boarded.name shouldBe "boarded"
        stringMap.name shouldBe "stringMap"
        numericMap.name shouldBe "numericMap"
        booleanMap.name shouldBe "booleanMap"
        survived.name shouldBe "survived"
      }
      "can be changed" in {
        val foo = (age + height).alias
        foo.name shouldBe "foo"
        val bar = (age / height).alias("bar")
        bar.name shouldBe "bar"
      }
    }
    "isRaw" should {
      "be true for raw features" in {
        height.isRaw shouldBe true
        weight.isRaw shouldBe true
      }
      "be false for non raw features" in {
        (height + weight).isRaw shouldBe false
        (weight / 2 + age + 1).isRaw shouldBe false
      }
    }
    "asRaw" should {
      "be a new raw feature of the same type" in {
        val f = weight / 2 + age
        val raw = f.asRaw()
        raw.isRaw shouldBe true
        raw should not be f
        raw.uid should not be f.uid
        raw.originStage.uid should not be f.originStage.uid
        raw.typeName shouldBe f.typeName
        raw.isResponse shouldBe f.isResponse
        raw.wtt.tpe =:= f.wtt.tpe shouldBe true
        raw.parents shouldBe Nil
      }
    }
    "isSubtypeOf" should {
      "correctly figure out feature subtypes" in {
        survived.isSubtypeOf[SingleResponse] shouldBe true
        boardedTime.isSubtypeOf[SingleResponse] shouldBe false
      }
    }
    "traverse" should {
      "operate correctly" in {
        (height + weight * age).traverse(List.empty[String])((acc, f) =>
          if (acc.contains(f.uid)) acc else f.uid :: acc
        ).length shouldBe 5
      }
      "collect raw features" in {
        height.rawFeatures shouldBe Seq(height)
        (height + weight).rawFeatures should contain theSameElementsAs Seq(height, weight)
      }
      "collect all features" in {
        height.allFeatures shouldBe Seq(height)
        val plus = height + weight
        plus.allFeatures should contain theSameElementsAs Seq(plus, height, weight)
      }
      "collect parent stages with correct distances" in {
        val plus = height + weight
        val mul = plus * age
        val div = mul / height
        div.parentStages() match {
          case Failure(e) => fail(e)
          case Success(stages) =>
            stages.size shouldBe 3
            stages.get(plus.originStage) shouldBe Some(2)
            stages.get(mul.originStage) shouldBe Some(1)
            stages.get(div.originStage) shouldBe Some(0)
        }
      }
    }
    "toString" should {
      "produce a nice string" in {
        val f = (height / 2) * weight + 1
        f.toString shouldBe
          s"Feature(name = ${f.name}, uid = ${f.uid}, " +
            s"isResponse = ${f.isResponse}, " +
            s"originStage = ${f.originStage.uid}, " +
            s"parents = ${f.parents.map(_.uid).mkString("[", ",", "]")}, " +
            s"distributions = ${f.distributions.map(_.toString).mkString("[", ",", "]")}, " +
            s"metadata = ${f.metadata})"
      }
    }
    "pretty parent stages" should {
      "print a single stage tree" in {
        height.prettyParentStages shouldBe "+-- SumRealNN(height)\n"
      }
      "print a multi stage tree" in {
        ((height / 2) * weight + 1).prettyParentStages shouldBe
          """|+-- plusS
             ||    +-- multiply
             ||    |    +-- SumReal(weight)
             ||    |    +-- divideS
             ||    |    |    +-- SumRealNN(height)
             |""".stripMargin
      }
    }
    "with distributions" should {
      "make a copy of the feature containing the specified distributions" in {
        val distrib1 = new FeatureDistributionLike {
          val count: Long = 1L
          val nulls: Long = 1L
          val distribution: Array[Double] = Array(0.5)
          val summaryInfo: Array[Double] = Array(0.5)
          val name: String = age.name
          val key: Option[String] = None
          val `type` = FeatureDistributionType.Training
        }
        val distrib2 = new FeatureDistributionLike {
          val count: Long = 1L
          val nulls: Long = 1L
          val distribution: Array[Double] = Array(0.5)
          val summaryInfo: Array[Double] = Array(0.5)
          val name: String = age.name
          val key: Option[String] = None
          val `type` = FeatureDistributionType.Scoring
        }
        age.distributions shouldBe Nil
        val newAge = age.withDistributions(Seq(distrib1, distrib2))
        newAge.name shouldEqual age.name
        newAge.isResponse shouldEqual age.isResponse
        newAge.originStage shouldEqual age.originStage
        newAge.parents shouldEqual age.parents
        newAge.uid shouldEqual age.uid
        newAge.distributions shouldBe Seq(distrib1, distrib2)
        newAge.trainingDistributions shouldBe Seq(distrib1)
        newAge.scoringDistributions shouldBe Seq(distrib2)
      }
    }
  }

}

