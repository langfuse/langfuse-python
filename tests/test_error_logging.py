import logging
import pytest

from langfuse.decorators.error_logging import (
    catch_and_log_errors,
    auto_decorate_methods_with,
)


# Test for the catch_and_log_errors decorator applied to a standalone function
@catch_and_log_errors
def function_that_raises():
    raise ValueError("This is a test error.")


def test_catch_and_log_errors_logs_error_silently(caplog):
    function_that_raises()

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    assert (
        "An error occurred in function_that_raises: This is a test error."
        in caplog.text
    )
    caplog.clear()


# Test for the auto_decorate_methods_with decorator applied to a class
@auto_decorate_methods_with(catch_and_log_errors, exclude=["excluded_instance_method"])
class TestClass:
    def instance_method(self):
        raise ValueError("Error in instance method.")

    def excluded_instance_method(self):
        raise ValueError("Error in instance method.")

    @classmethod
    def class_method(cls):
        raise ValueError("Error in class method.")

    @staticmethod
    def static_method():
        raise ValueError("Error in static method.")


def test_auto_decorate_class_methods(caplog):
    test_obj = TestClass()

    # Test the instance method
    test_obj.instance_method()
    assert "Error in instance method." in caplog.text
    caplog.clear()

    # Test the class method
    TestClass.class_method()
    assert "Error in class method." in caplog.text
    caplog.clear()

    # Test the static method
    TestClass.static_method()
    assert "Error in static method." in caplog.text
    caplog.clear()

    # Test the excluded instance method that should raise an error
    with pytest.raises(ValueError, match="Error in instance method."):
        test_obj.excluded_instance_method()
