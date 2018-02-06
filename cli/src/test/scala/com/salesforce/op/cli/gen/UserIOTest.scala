/*
 * Copyright (c) 2017, Salesforce.com, Inc.
 * All rights reserved.
 */

package com.salesforce.op.cli.gen

import com.salesforce.op.test.TestCommon
import org.scalatest.{Assertions, FlatSpec}

/**
 * Test for Console methods.
 */
class UserIOTest extends FlatSpec with TestCommon with Assertions {


  private case class Oracle(answers: String*) extends UserIO {
    private var i = -1
    var question = "---"

    override def readLine(q: String): Option[String] = {
      question = q
      i += 1
      if (i < answers.length) Some(answers(i)) else throw new IllegalStateException(s"Out of answers, q=$q")
    }
  }

  Spec[UserIO] should "do qna" in {
    // @see https://www.urbandictionary.com/define.php?term=aks
    def aksme(q: String, answers: String*): Option[String] = {
      Oracle(answers: _*).qna(q, _.length == 1, Map("2*3" -> "6", "3+2" -> "5"))
    }

    aksme("2+2", "11", "22", "?") shouldBe Some("?")
    aksme("2+2", "4", "5", "?") shouldBe Some("4")
    aksme("2+3", "44", "", "?") shouldBe Some("?")
    aksme("2*3", "4", "?") shouldBe Some("6")
    aksme("3+2", "4", "?") shouldBe Some("5")
  }

  it should "ask" in {

    // @see https://www.urbandictionary.com/define.php?term=aks
    def aksme[Int](q: String, opts: Map[Int, List[String]], answers: String*): (String, Int) = {
      val console = Oracle(answers: _*)
      val answer = console.ask(q, opts) getOrElse fail(s"A problem answering question $q")
      (console.question, answer)
    }

    an[IllegalStateException] should be thrownBy
      aksme("what is your name?", Map(1 -> List("one", "uno")), "11", "1", "?")

    aksme("what is your name?",
      Map(
        1 -> List("Nessuno", "Nobody"),
        2 -> List("Ishmael", "Gantenbein")),
      "5", "1", "?") shouldBe("what is your name? [0] Nessuno [1] Ishmael: ", 2)
  }

}
