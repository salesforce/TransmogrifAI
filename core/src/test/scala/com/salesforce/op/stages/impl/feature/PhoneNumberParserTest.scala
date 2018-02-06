/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.stages.impl.feature

import com.salesforce.op.OpWorkflow
import com.salesforce.op.features.FeatureLike
import com.salesforce.op.features.types._
import com.salesforce.op.stages.base.binary.BinaryTransformer
import com.salesforce.op.test._
import com.salesforce.op.testkit.RandomText
import com.salesforce.op.utils.spark.RichDataset._
import org.apache.spark.sql
import org.apache.spark.sql.Row
import org.junit.runner.RunWith
import org.scalatest.FlatSpec
import org.scalatest.junit.JUnitRunner


@RunWith(classOf[JUnitRunner])
class PhoneNumberParserTest extends FlatSpec with TestSparkContext {

  val userDefParser = new ParsePhoneNumber()
  val namesAndCode = Map("US" -> "United States", "CA" -> "Canada", "ZW" -> "Zimbabwe")
  userDefParser.setCodesAndCountries(namesAndCode)

  private val regionCodes = namesAndCode.keys.map(_.toUpperCase()).toArray
  private val regionNames = namesAndCode.values.map(_.toUpperCase()).toArray
  private val defaultCodes = PhoneNumberParser.DefaultCountryCodes.keys.map(_.toUpperCase).toArray
  private val defaultNames = PhoneNumberParser.DefaultCountryCodes.values.map(_.toUpperCase).toArray

  val defaultParser = new ParsePhoneNumber
  val defaultValid = new IsValidPhoneNumber
  val defaultUnaryParse = new ParsePhoneDefaultCountry()
  val defaultUnaryValid = new IsValidPhoneDefaultCountry()

  val pns = Array(Some("+15105556666"), Some("510 555 6666"), Some("+1+3456"), Some("+1510334455667788"), None)
  val answerParse = Array(Some("+15105556666"), Some("+15105556666"), None, Some("+15103344556"), None)
  val answerValid = Array(Some(true), Some(true), None, Some(true), None)

  val (goodPhones, pGood) = TestFeatureBuilder(RandomText.phones.take(1000).toSeq)
  val (badPhones, pBad) = TestFeatureBuilder(RandomText.phonesWithErrors(1.0).take(1000).toSeq)

  Spec(PhoneNumberParser.getClass) should "should properly clean invalid tokens from phone numbers" in {
    val allAscii = (32 to 126).map(_.toChar).foldLeft("")((s, x) => s + x)
    PhoneNumberParser.cleanNumber(allAscii) shouldBe "+0123456789"
  }

  it should "be a BinaryTransformer" in {
    userDefParser shouldBe a[BinaryTransformer[_, _, _]]
  }

  it should "fail when setting invalid region code" in {
    intercept[IllegalArgumentException] {
      userDefParser.setCodesAndCountries(Map("foo" -> "bar"))
    }
  }

  it should "use default regions" in {
    PhoneNumberParser.validCountryCode(Phone(""), Text("AF"), PhoneNumberParser.DefaultRegion,
      defaultCodes, defaultNames) shouldBe "AF"
  }

  it should "return default region if country code is not found" in {
    PhoneNumberParser.validCountryCode(Phone(""), Text("FooBar"), PhoneNumberParser.DefaultRegion,
      Array.empty, Array.empty) shouldBe
      defaultParser.getOrDefault(defaultParser.defaultRegion)
  }

  it should "return a close match if country code is not found or country is misspelled" in {
    val countries = Array("uS", "United St America", "States of America", "Grece", "Switzland", "USA").map(Text(_))
    countries.map(PhoneNumberParser.validCountryCode(Phone(""), _, PhoneNumberParser.DefaultRegion,
      defaultCodes, defaultNames)) shouldBe Array("US", "US", "US", "GR", "CH", "US")
  }

  it should "country code should always be generic international locale if number starts with +" in {
    PhoneNumberParser.validCountryCode(Phone("+1234566"), Text("CN"), PhoneNumberParser.DefaultRegion,
      defaultCodes, defaultNames) shouldBe PhoneNumberParser.InternationalCode
  }

  it should "find valid country code by string match" in {
    val countries = Array("uS", "CD", "United", "Zimbwe", "USA").map(Text(_))
    countries.map(PhoneNumberParser.validCountryCode(Phone(""), _,
      PhoneNumberParser.DefaultRegion, regionCodes, regionNames)) should contain theSameElementsInOrderAs
      Array("US", "CD", "US", "ZW", "US")
  }

  it should "find closest valid match in input or supported if input region is not part of the set" in {
    // valid region
    val result = PhoneNumberParser.validCountryCode(Phone(""), Text("AF"),
      PhoneNumberParser.DefaultRegion, regionCodes, regionNames)
    assert(PhoneNumberParser.phoneUtil.getSupportedRegions.contains(result))

    // not valid
    val result2 = PhoneNumberParser.validCountryCode(Phone(""), Text("FOo"),
      PhoneNumberParser.DefaultRegion, regionCodes, regionNames)
    assert(regionCodes.toSet.contains(result2))
  }

  "International Code" should "be ZZ" in {
    PhoneNumberParser.InternationalCode shouldBe "ZZ"
  }

  Spec[ParsePhoneNumber] should "have properly set default param settings" in {
    defaultParser.getOrDefault(defaultParser.regionCodes) shouldBe defaultCodes
    defaultParser.getOrDefault(defaultParser.countryNames) shouldBe defaultNames
    defaultParser.getOrDefault(defaultParser.defaultRegion) shouldBe "US"
    defaultParser.getOrDefault(defaultParser.strictValidation) shouldBe false
  }

  it should "set country code and country name correctly" in {
    userDefParser.get(userDefParser.regionCodes).get should contain theSameElementsAs regionCodes
    userDefParser.get(userDefParser.countryNames).get should contain theSameElementsAs regionNames
  }

  it should "parse US phone numbers" in {
    pns.map(x => userDefParser.transformFn(Phone(x), Text("US")).value) should
      contain theSameElementsInOrderAs answerParse
  }

  it should "return None is input phone number is None" in {
    userDefParser.transformFn(Phone.empty, Text.empty) shouldBe Phone.empty
  }

  it should "parse based on strictness setting" in {
    // valid after truncating
    val pn = "+1510334455667788"
    val local = new ParsePhoneNumber()
    local.transformFn(Phone(pn), Text("US")) shouldBe Phone("+15103344556")

    local.setStrictness(true)
    local.transformFn(Phone(pn), Text("US")) shouldBe Phone(None)
  }

  it should "find all numbers less than length of 2 to be invalid independent of country" in {
    val pns = Array("", "5")
    val answer = Array(None, None)
    pns.map(x => userDefParser.transformFn(Phone(x), Text.empty).value) should
      contain theSameElementsInOrderAs answer
  }

  Spec[IsValidPhoneNumber] should "validate based on strictness setting" in {
    // valid after truncating
    val pn = "+1510334455667788"
    val local = new IsValidPhoneNumber()
    local.transformFn(Phone(pn), Text("US")) shouldBe Binary(true)
    local.transformFn(Phone(None), Text("US")) shouldBe Binary(None)

    local.setStrictness(true)
    local.transformFn(Phone(pn), Text("US")) shouldBe Binary(false)
  }
  it should "return None is input phone number is None" in {
    defaultValid.transformFn(Phone.empty, Text.empty) shouldBe Phone.empty
  }
  it should "find all numbers less than length of 2 to be invalid independent of country" in {
    val pns = Array("", "5")
    val answer = Array(None, None)
    pns.map(x => defaultValid.transformFn(Phone(x), Text.empty).value) should
      contain theSameElementsInOrderAs answer
  }
  it should "validate US phone numbers" in {
    pns.map(x => defaultValid.transformFn(Phone(x), Text("US")).value) should
      contain theSameElementsInOrderAs answerValid
  }

  Spec[ParsePhoneDefaultCountry] should "validate US phone numbers without a country code" in {
    val unaryPhoneValidator = new ParsePhoneDefaultCountry()
    pns.map(x => unaryPhoneValidator.transformFn(Phone(x)).value) should
      contain theSameElementsInOrderAs answerParse
  }
  it should "need a country identifyer when the local does not match the default" in {
    val unaryPhoneValidator = new ParsePhoneDefaultCountry().setStrictness(false).setDefaultRegion("ZW")
    pns.map(x => unaryPhoneValidator.transformFn(Phone(x)).value) should
      contain theSameElementsInOrderAs Array(Some("+15105556666"), None, None, Some("+15103344556"), None)
  }
  it  should "parse based on strictness setting" in {
    // valid after truncating
    val pn = "+1510334455667788"
    val local = new ParsePhoneDefaultCountry()
    local.transformFn(Phone(pn)) shouldBe Phone("+15103344556")

    local.setStrictness(true)
    local.transformFn(Phone(pn)) shouldBe Phone(None)
  }
  it should "return None is input phone number is None" in {
    defaultUnaryParse.transformFn(Phone.empty) shouldBe Phone.empty
  }
  it should "return valid phones numbers with shortcut" in {
    val (ds, pn, cc) = TestFeatureBuilder(Seq[(Phone, Text)]((Phone("5105556666"), Text("US"))))
    val result = pn.parsePhone(cc)

    result.name shouldBe result.originStage.outputName
    result.parents shouldBe Array(pn, cc)
    result.originStage shouldBe a[ParsePhoneNumber]

    val data = result.originStage.asInstanceOf[ParsePhoneNumber].transform(ds)
    val ans = data.take(1, result)
    ans(0) shouldBe Phone("+15105556666")
  }
  it should "correctly parse valid phone numbers with shortcut" in {
    val (ds, pn) = TestFeatureBuilder(Seq[Phone](Phone("5105556666"), Phone("99995105556666"), Phone.empty))
    val result = pn.parsePhoneDefaultCountry()

    result.name shouldBe result.originStage.outputName
    result.parents shouldBe Array(pn)
    result.originStage shouldBe a[ParsePhoneDefaultCountry]

    val data = new OpWorkflow().setResultFeatures(result).transform(ds)
    val ans = data.take(3, result)

    result.name shouldBe result.originStage.outputName
    ans should contain theSameElementsInOrderAs Array(Phone("+15105556666"), Phone(None), Phone.empty)
  }

  it should "correctly parse valid phone numbers with shortcut on a random sample" in {
    val result: FeatureLike[Phone] = pGood.parsePhoneDefaultCountry()
    val data: sql.DataFrame = new OpWorkflow().setResultFeatures(result).transform(goodPhones)
    val pps: Array[Row] = data.select(result).collect()

    import com.salesforce.op.utils.spark.RichRow._

    def check(r: Row): Unit = {
      val phone: Phone = r.getFeatureType(result)
      withClue(s"From $r") { phone.isEmpty shouldBe false }
    }

    pps foreach check
  }
  it should "skip parse invalid phone numbers with shortcut on a random sample" in {
    val result = pBad.parsePhoneDefaultCountry()
    val data = new OpWorkflow().setResultFeatures(result).transform(badPhones)
    data.collect(result).forall(_.isEmpty) shouldBe true
  }

  Spec[IsValidPhoneDefaultCountry] should "validate based on strictness setting" in {
    // valid after truncating
    val pn = "+1510334455667788"
    val local = new IsValidPhoneDefaultCountry()
    local.transformFn(Phone(pn)) shouldBe Binary(true)
    local.transformFn(Phone(None)) shouldBe Binary(None)

    local.setStrictness(true)
    local.transformFn(Phone(pn)) shouldBe Binary(false)
  }
  it should "return None is input phone number is None" in {
    defaultUnaryValid.transformFn(Phone.empty) shouldBe Phone.empty
  }
  it should "find all numbers less than length of 2 to be invalid independent of country" in {
    val pns = Array("", "5")
    val answer = Array(None, None)
    pns.map(x => defaultUnaryValid.transformFn(Phone(x)).value) should contain theSameElementsInOrderAs answer
  }
  it should "validate US phone numbers" in {
    pns.map(x => defaultUnaryValid.transformFn(Phone(x)).value) should contain theSameElementsInOrderAs answerValid
  }
  it should "validate phone numbers with isValid shortcut" in {
    val (ds, pn, cc) = TestFeatureBuilder(Seq[(Phone, Text)]((Phone("5105556666"), Text("US"))))
    val result = pn.isValidPhone(cc, namesAndCode)

    result.name shouldBe result.originStage.outputName
    result.parents shouldBe Array(pn, cc)
    result.originStage shouldBe a[IsValidPhoneNumber]

    val data = result.originStage.asInstanceOf[IsValidPhoneNumber].transform(ds)
    val ans = data.take(1, result)
    ans(0) shouldBe Binary(true)
  }
  it should "correctly identify valid phone numbers with shortcut" in {
    val (ds, pn) = TestFeatureBuilder(Seq[Phone](Phone("5105556666"), Phone("99995105556666"), Phone.empty))
    val result = pn.isValidPhoneDefaultCountry()

    result.name shouldBe result.originStage.outputName
    result.parents shouldBe Array(pn)
    result.originStage shouldBe a[IsValidPhoneDefaultCountry]

    val data = new OpWorkflow().setResultFeatures(result).transform(ds)
    val ans = data.take(3, result)

    result.name shouldBe result.originStage.outputName
    ans should contain theSameElementsInOrderAs Array(Binary(true), Binary(false), Binary.empty)
  }
  it should "correctly identify valid phone numbers on a random sample" in {
    val result = pGood.isValidPhoneDefaultCountry()
    val data = new OpWorkflow().setResultFeatures(result).transform(goodPhones)
    data.collect(result).forall(_.toDouble() == 1.0) shouldBe true
  }
  it should "correctly identify invalid phone numbers on a random sample" in {
    val result = pBad.isValidPhoneDefaultCountry()
    val data = new OpWorkflow().setResultFeatures(result).transform(badPhones)
    data.collect(result).forall(_.toDouble() == 0.0) shouldBe true
  }


}

