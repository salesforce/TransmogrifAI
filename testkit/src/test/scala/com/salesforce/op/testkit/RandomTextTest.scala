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

package com.salesforce.op.testkit

import com.salesforce.op.features.types.Text
import com.salesforce.op.test.TestCommon
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Assertion, Assertions, FlatSpec}

import scala.language.postfixOps
import scala.util.Random


@RunWith(classOf[JUnitRunner])
class RandomTextTest extends FlatSpec with TestCommon with Assertions {
  val numTries = 1000
  private val rngSeed = 2718281828314L

  private def check[T <: Text](
    g: RandomText[T],
    condition: String => Boolean = _ => true,
    sampleSize: Int = 0,
    expected: List[String] = Nil
  ) = {
    g reset rngSeed
    val probabilityOfEmpty: Double = g.probabilityOfEmpty

    def segment = g limit numTries

    val numberOfEmpties = segment count (_.isEmpty)

    val expectedNumberOfEmpties = probabilityOfEmpty * numTries
    withClue(s"numEmpties = $numberOfEmpties, expected $expectedNumberOfEmpties") {
      math.abs(numberOfEmpties - expectedNumberOfEmpties) < 2 * math.sqrt(numTries) shouldBe true
    }

    val maybeStrings = segment filterNot (_.isEmpty) map (_.value)
    val generated = maybeStrings collect { case Some(s) => s } toSet

    withClue(s"number of distinct strings = ${generated.size}, expected:") {
      math.abs(maybeStrings.size - generated.size) < maybeStrings.size / 20
    }

    val badOnes = generated filterNot condition
    badOnes shouldBe empty
    checkSample(g limit sampleSize, expected)
  }

  def checkSample[T <: Text](sample: List[T], expected: List[String]): Assertion = {
    val actual = sample map (_.value) collect { case Some(e) => e }
    // the following line just generates a chunk of code for your test if it fails
    if (actual != expected) println(actual.mkString("      \"", "\",\n      \"", "\"\n"))
    actual shouldBe expected
  }

  "Text generator" should "generate Empties and distinct strings" in {
    check(RandomText.strings(42, 300) withProbabilityOfEmpty 0.5)

    check(RandomText.strings(300, 300) withProbabilityOfEmpty 0.1)
  }

  "Text generator" should "generate strings of varying length" in {
    val g = RandomText.strings(10, 1000) withProbabilityOfEmpty 0.001
    check(g, (s: String) => s.length >= 10 && s.length <= 1000)
  }

  "TextArea generator" should "generate textareas" in {
    val g = RandomText.textAreas(40, 50) withProbabilityOfEmpty 0.001
    check(g, (s: String) => s.length >= 40 && s.length <= 50)
  }

  "textFromDomain generator" should "generate Texts from provided domain" in {
    val dom = List("Alpha", "Beta", "Gamma", "Delta", "Epsilon")
    val g = RandomText.textFromDomain(dom) withProbabilityOfEmpty 0.001
    check(g, dom.toSet)
  }

  "textAreaFromDomain generator" should "generate TextAreas from provided domain" in {
    val dom = List("Alpha", "Beta", "Gamma", "Delta", "Epsilon")
    val g = RandomText.textAreaFromDomain(dom) withProbabilityOfEmpty 0.001
    check(g, dom.toSet)
  }

  "PickList generator" should "generate picklists" in {
    val dom = List("Red", "Green", "Blue")
    val g = RandomText.pickLists(dom) withProbabilityOfEmpty 0.001
    check(g, dom.toSet)
  }

  "PickList generator with distribution" should "generate picklists properly" in {
    val dom = List("Red", "Green", "Blue")
    val g0 = RandomText.pickLists(dom, List(0.5, 0.8, 1.0))
    val g = g0 withProbabilityOfEmpty 0.001
    check(g, dom.toSet)
    val sample = g limit 10

    val expected = List(
      "Red",
      "Red",
      "Blue",
      "Blue",
      "Green",
      "Red",
      "Blue",
      "Red",
      "Red",
      "Blue"
    )
    checkSample(sample, expected)
  }

  "ComboBox generator" should "generate comboboxes" in {
    val dom = List("Male", "Female", "VIM user")
    check(RandomText.comboBoxes(dom) withProbabilityOfEmpty 0.001, dom.toSet)
  }

  // scalastyle:off
  "random ComboBox generator" should "generate totally random comboboxes" in {
    check(RandomText.randomComboBoxes, sampleSize = 3, expected = List(
      "Ŧ䑮ⶬ봫缭៨╻䀹谶雮逘ᗯ",
      "꥟邐䶭ɍ槎⬋㋴㒍墏冀",
      "뛇笅ₑ㣏"
    ))
  }
  // scalastyle:on

  "base64 generator" should "generate base64" in {
    val g = RandomText.base64(10, 15) withProbabilityOfEmpty 0.001
    check(g, (s: String) => s.length >= 10 && s.length <= 20)
  }

  "phone generator" should "generate phones like in the movies" in {
    val g = RandomText.phones withProbabilityOfEmpty 0.001
    check(g,
      (s: String) => s.length == 10 && s.substring(3, 6) == "555")
  }

  "phone generator with errors" should "generate phones, failing at times" in {
    val g = RandomText.phonesWithErrors(0.8)
    check(g,
      sampleSize = 6,
      expected = List(
        "10188613",
        "10296495",
        "10377823",
        "1956951",
        "060860",
        "113876634"
      ))
  }

  "postal code generator" should "generate zipcodes" in {
    val g = RandomText.postalCodes withProbabilityOfEmpty 0.07
    check(g,
      (s: String) => {
        val n = s.toInt
        n < 100000 && n >= 1000
      })
  }

  "id generator" should "generate IDs" in {
    val g = RandomText.ids withProbabilityOfEmpty 0.001
    check(g,
      (s: String) => s.length > 0 && s.length < 60 && (s matches "\\w+"))
  }

  "id generator" should "not produce empty IDs by default" in {
    val g = RandomText.ids.limit(10000)
    g.forall(id => !id.value.isEmpty && !id.value.contains("")) shouldBe true
  }

  "unique id generator" should "generate IDs" in {
    val g = RandomText.uniqueIds
    check(g,
      (s: String) => s.length > 0 && s.length < 60 && (s matches "\\w+_[0-9a-f]+"),
      sampleSize = 10,
      expected = List(
        "RyYaZ9pXqz3hwK_7d2",
        "_ReDX_7d3",
        "6Ns4ptuMoUUTuEQ_7d4",
        "mAs_7d5",
        "3L0zHp5643SIKnZS_7d6",
        "6ZB_7d7",
        "aRDrgHMuzPM_7d8",
        "U3_7d9",
        "cI7va5lYQ_7da",
        "rSUkRzxMgIYVa_7db"
      )
    )
  }

  "unique id generator" should "not produce empty IDs by default" in {
    val g = RandomText.uniqueIds.limit(10000)
    g.forall(id => !id.value.isEmpty && !id.value.contains("")) shouldBe true
  }

  "emails generator" should "generate emails" in {
    val g = RandomText.emails("whitehouse.gov") withProbabilityOfEmpty 0.1
    check(g, sampleSize = 10,
      expected = List(
        "dlangdon@whitehouse.gov", // https://github.com/dlangdon
        "Johnnie.Garvin@whitehouse.gov", // https://www.youtube.com/watch?v=L2qZfZMXZqU
        "Teri.Ordonez@whitehouse.gov", // https://www.facebook.com/teri.ordonez
        "Ping.Rosa@whitehouse.gov", // https://www.facebook.com/ping.rosa
        "jlovell195@whitehouse.gov", // https://github.com/jlovell
        "Ahmet.Casillas@whitehouse.gov", // https://kiwi.qa/e610ae76184bd3
        "swoodward484@whitehouse.gov",
        "jcoburn578@whitehouse.gov", // https://twitter.com/coburnj
        "mlogsdon723@whitehouse.gov",
        "Sridhar.Sadler@whitehouse.gov"
      )
    )
  }

  "emails generator" should "generate emails in a variety of domains" in {
    val domains = RandomStream of List("leha.im", "noxep.com") distributedAs List(0.7, 1.0)

    val g = RandomText.emailsOn(domains)
    check(g, sampleSize = 10,
      expected = List(
        "Cory.Kozlowski@leha.im", // http://corykozlowski.ca/
        "Pravin.Sparks@noxep.com", // https://plus.google.com/108597651305169361830
        "jdawkins461@leha.im", // http://jdawkins.co.uk/
        "osylvester@leha.im", // https://oliviacsylvester.wordpress.com/author/osylvester/
        "Klaus.Bayer@noxep.com", // http://nbkterracotta.com/author/klaus-bayer/
        "mmoseley@leha.im", // https://www.pinterest.com/mmoseley/
        "Stanislaw.Cone@noxep.com",
        "Loren.Roden@leha.im", // https://www.findagrave.com/cgi-bin/fg.cgi?page=gr&GRid=26629192
        "Winnie.Conway@leha.im", // http://freepages.family.rootsweb.ancestry.com/~pastancestors/conways.jpg
        "jbueno749@noxep.com"   // https://twitter.com/JBueno_UKCP
      )
    )
  }

  "urls generator" should "generate urls" in {
    val g = RandomText.urls withProbabilityOfEmpty 0.1
    check(g, sampleSize = 10,
      expected = List(
        "https://efd6?Lan=%C2%88%E7%B6%94&Mic=%E3%97%87%E1%BF%B6",
        "https://3yx.af4h.za4w?Geo=%E7%AE%B4%E3%84%B8&Tam=%E0%BF%A5%EC%9B%A0&Cyn=%E7%AB%B3%E4%A9%BB",
        "http://ddp3.f4te?Cat=%E4%9D%98%E0%BF%A9&Doy=%E4%AA%AA%E4%8A%AE&Hor=%E5%80%A4%E6%9C%B4",
        "http://7x3?Hug=%E9%AD%A0%EB%90%8B&Cla=%E3%BC%BE%E8%94%BB",
        "https://8hs?Sal=%E9%9D%81%E2%8E%B1&Bra=%E4%91%B1%E8%86%A5&Lin=%E9%86%B9%E1%BB%B9",
        "https://8rq?Edu=%E1%B7%B6%EA%B4%84&Her=%EB%8B%B1%E3%AB%A5&Car=%D8%BE%E9%9C%8E",
        "https://aw8x.4hr.xfw?Lou=%E3%BF%8F%EA%AE%8A",
        "https://579?Ada=%D1%87%E4%BB%8F&Pla=%EA%8A%81%E9%B8%A3",
        "https://f67y.znzs.vk98?Rox=%E1%BD%B5%E1%96%BA",
        "https://u62f.kkf2.w81?Vic=%E0%B4%86%E6%8D%94&Cha=%C6%B8%EB%87%9E&Sta=%EA%90%A5%EA%B5%96"
      )
    )
  }

  // scalastyle:off
  "urls generator" should "generate urls in a variety of domains" in {
    val domains = RandomStream of List("myredbook.com", "bbc.co.uk") distributedAs List(0.7, 1.0)

    val g = RandomText.urlsOn(domains)
    check(g, sampleSize = 6,
      expected = List(
        "https://myredbook.com?Eli=%E6%B8%87%40",
        "https://myredbook.com?Chr=%E5%95%A9%E7%B1%85&Raj=%E7%B5%89%EC%AE%A1&Hir=%E5%B3%8F%E0%B4%A3",
        "https://myredbook.com?Bil=%E3%A5%B6%E2%80%87&Bet=%E1%A2%81%EA%9C%83",
        "https://bbc.co.uk?Deb=%EB%A6%A7%E6%9C%96&Pam=%E2%94%9A%E9%B1%91&Mas=%E4%B7%91%E0%B8%84",
        "https://myredbook.com?Cla=%E9%99%B9%E4%8A%93&Cha=%E3%95%98%EA%A3%A7&Eve=%EC%91%90%E8%87%B1",
        "https://myredbook.com?Lea=%E2%BC%84%EB%91%A3&Mur=%E2%83%BD%E1%92%83"
      )
    )
  }
  // scalastyle:on

  // scalastyle:off
  "countries generator" should "generate names of countries" in {
    val g = RandomText.countries withProbabilityOfEmpty 0.3
    check(g, sampleSize = 15,
      expected = List(
        "Freedonia",
        "Somalia",
        "Pottsylvania",
        "São Rico",
        "Qumar",
        "South Sudan",
        "Graznavia",
        "Illyria",
        "Santo Marco"
      ))
  }
  // scalastyle:on

  "states generator" should "generate names of states" in {
    val g = RandomText.states withProbabilityOfEmpty 0.3
    check(g, sampleSize = 10,
      expected = List(
        "Indiana",
        "Idaho",
        "Montana",
        "Alabama",
        "New Hampshire",
        "Indiana",
        "New Jersey"
    ))
  }

  "cities generator" should "generate names of cities" in {
    val g = RandomText.cities withProbabilityOfEmpty 0.2
    check(g, sampleSize = 10,
      expected = List(
        "Dorris",
        "Gridley",
        "Rio Dell",
        "San Leandro",
        "Chico",
        "Willits",
        "Daly City",
        "Concord",
        "Berkeley"
    ))
  }

  "streets generator" should "generate names of streets" in {
    val g = RandomText.streets withProbabilityOfEmpty 0.2
    check(g, sampleSize = 10,
      expected = List(
        "Lester Avenue",
        "Dry Creek Road",
        "Mission Street",
        "Berryessa Road",
        "Alum Rock Avenue",
        "Delmas Avenue",
        "Cunningham Avenue",
        "Park Avenue",
        "Washington Street"
    ))
  }
}
