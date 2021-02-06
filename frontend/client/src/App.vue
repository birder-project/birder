<template>
  <div id="app">
    <RouterView />
  </div>
</template>

<script>
import mmsManagementActionType from '@/store/modules/mmsManagement/action-types';

export default {
  components: {},

  data() {
    return {};
  },

  created() {
    this.$store.dispatch(mmsManagementActionType.getModels);
  },
};
</script>

<style lang="scss">
@import 'bulma';
</style>
