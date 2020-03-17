
# TODO: Add functions to stop fleet, terminate fleet, terminate finished instances, check if instances are finished
class Fleet:
    def __init__(self, fleet_name, instance_config_list, session):
        self.fleet_name = fleet_name
        self.instance_config_list = instance_config_list
        self.session = session

    @property
    def fleet_size(self):
        return len(self.instance_config_list)

    @property
    def fleet_size_active(self):
        size_active = 0
        for instance in self.instance_config_list:
            if instance.is_created:
                size_active += 1
        return size_active

    @property
    def fleet_size_running(self):
        size_running = 0
        for instance in self.instance_config_list:
            if instance.is_running:
                size_running += 1
        return size_running

    def create_fleet(self):
        for instance in self.instance_config_list:
            instance.create_instance(session=self.session)

    # TODO: Check if terminated/stopped, if so it currently stalls the whole thing and everything is messed up.
    def run_fleet(self):
        results = []
        for instance in self.instance_config_list:
            result = instance.run_instance(session=self.session)
            results.append(result)
        return results

    # TODO: Sort alphabetically
    def get_fleet_info(self):
        ec2 = self.session.client('ec2')
        fleet_info = ec2.describe_instances(
            Filters=[
                {
                    'Name': 'tag:fleet',
                    'Values': [
                        self.fleet_name,
                    ]
                }
            ]
        )
        fleet_info = fleet_info['Reservations']
        return fleet_info

    def print_info(self):
        print('Fleet Info:')
        print('\tFleet Name: %s' % self.fleet_name)
        print('\tFleet Size: %s' % self.fleet_size)
        print('\tFleet Size Active: %s' % self.fleet_size_active)
        print('\tFleet Size Running: %s' % self.fleet_size_running)

    def __str__(self):
        return str(self.__dict__)
