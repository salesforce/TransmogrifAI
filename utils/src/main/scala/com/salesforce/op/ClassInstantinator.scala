package com.salesforce.op
import scala.util.Try

/**
  * @author ksuchanek
  * @since 214
  */
object ClassInstantinator {
  private[this] val ClassOfString = classOf[String]
  private[this] val ClassOfInt = classOf[Int]

  def instantinateRaw(className: String, args: Array[AnyRef]): Try[Any] = {
    Try {
      val clazz = getClass.getClassLoader.loadClass(className)
      val constructor = clazz.getConstructors.head
      constructor.newInstance(args: _*)
    }
  }

  def instantinate(className: String, args: Array[String]): Try[Any] = {

    Try {
      val clazz = getClass.getClassLoader.loadClass(
        //"com.salesforce.op.features.FeatureBuilder$$anonfun$fromRow$1"
        className
      )
      val constructor = clazz.getConstructors.head
      val paramsType: Array[Class[_]] = constructor.getParameterTypes
      if (paramsType.length != args.length) {
        throw new IllegalArgumentException(
          s"Constructor's parameters count mismatch: ${paramsType.length} ${args.length}"
        )
      }

      val typedArgs = new Array[AnyRef](args.length)

      var i = 0
      println(paramsType.mkString(","))
      while (i < args.length) {
        typedArgs(i) = paramsType(i) match {
          case ClassOfString => args(i).toString
          case ClassOfInt    => Int.box(args(i).toInt)
          case x =>
            throw new IllegalArgumentException(
              s"Unsupported type: ${x.getName}"
            )

        }
        i += 1
      }

      constructor.newInstance(typedArgs: _*)

    }
  }
}
