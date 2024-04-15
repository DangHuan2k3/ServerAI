from siteController import SiteController

# Example usage:
singleton1 = SiteController()
singleton2 = SiteController()

print(singleton1 is singleton2)  # Output will be True
